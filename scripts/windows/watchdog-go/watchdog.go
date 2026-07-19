package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

type cycleResult struct {
	Desktop     string `json:"desktop"`
	Backend     string `json:"backend"`
	BackendPID  uint32 `json:"backendPid,omitempty"`
	BackendPort int    `json:"backendPort,omitempty"`
}

type WatchdogState struct {
	UpdatedAt               string       `json:"updatedAt"`
	WatchdogPID             int          `json:"watchdogPid"`
	Paused                  bool         `json:"paused"`
	Result                  cycleResult  `json:"result"`
	ConsecutiveBackendFails int          `json:"consecutiveBackendFails"`
	PackagedExe             string       `json:"packagedExe,omitempty"`
	ListenAddr              string       `json:"listenAddr,omitempty"`
	TsnetHostname           string       `json:"tsnetHostname,omitempty"`
	TsnetEnabled            bool         `json:"tsnetEnabled"`
}

type Watchdog struct {
	cfg    Config
	logger *Logger
	back   *BackendManager

	mu        sync.RWMutex
	paused    bool
	failCount int
	lastState WatchdogState
}

func NewWatchdog(cfg Config, logger *Logger) *Watchdog {
	return &Watchdog{
		cfg:    cfg,
		logger: logger,
		back:   NewBackendManager(cfg, logger),
		lastState: WatchdogState{
			WatchdogPID: os.Getpid(),
			PackagedExe: cfg.PackagedExe,
			ListenAddr:  cfg.ListenAddr,
			TsnetHostname: cfg.TsnetHostname,
			TsnetEnabled:  cfg.EnableTsnet && cfg.TsAuthKey != "",
		},
	}
}

func (w *Watchdog) PrewarmBackend() {
	if !w.cfg.PrewarmBackend {
		return
	}
	if _, err := w.back.EnsureHealthy(); err != nil {
		w.logger.Infof("prewarm backend: %v", err)
	}
}

func (w *Watchdog) findAnyHealthyBackend() *backendInfo {
	if child := findHealthyDesktopBackend(); child != nil {
		return child
	}
	if managed := w.back.currentHealthy(); managed != nil {
		return managed
	}
	return loadManifestBackend(w.cfg)
}

func (w *Watchdog) State() WatchdogState {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.lastState
}

func (w *Watchdog) SetPaused(v bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.paused = v
	w.lastState.Paused = v
}

func (w *Watchdog) IsPaused() bool {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.paused
}

func (w *Watchdog) saveState(result cycleResult) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.lastState = WatchdogState{
		UpdatedAt:               time.Now().Format(time.RFC3339),
		WatchdogPID:             os.Getpid(),
		Paused:                  w.paused,
		Result:                  result,
		ConsecutiveBackendFails: w.failCount,
		PackagedExe:             w.cfg.PackagedExe,
		ListenAddr:              w.cfg.ListenAddr,
		TsnetHostname:           w.cfg.TsnetHostname,
		TsnetEnabled:            w.cfg.EnableTsnet && w.cfg.TsAuthKey != "",
	}
	raw, err := json.MarshalIndent(w.lastState, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(w.cfg.StatePath, raw, 0o644)
}

func (w *Watchdog) RunCycle() cycleResult {
	if w.IsPaused() {
		res := cycleResult{Desktop: "paused", Backend: "paused"}
		w.saveState(res)
		return res
	}

	desktop, derr := getDesktopProcesses()
	if w.cfg.PrewarmBackend {
		if _, err := w.back.EnsureHealthy(); err != nil {
			w.logger.Infof("ensure managed backend: %v", err)
		}
	}
	backend := w.findAnyHealthyBackend()

	if derr != nil || len(desktop) == 0 {
		var skipPID uint32
		if managed := w.back.currentHealthy(); managed != nil {
			skipPID = managed.PID
		}
		stopOrphanDesktopBackends(w.logger, w.cfg, skipPID)
		w.logger.Infof("Desktop DOWN — relaunch")
		startPackagedDesktop(w.cfg, w.logger, w.back)
		w.mu.Lock()
		w.failCount = 0
		w.mu.Unlock()
		res := cycleResult{Desktop: "relaunched", Backend: "pending"}
		w.saveState(res)
		return res
	}

	if backend == nil {
		w.logger.Infof("Desktop UP but backend DOWN — starting managed serve")
		if _, err := w.back.EnsureHealthy(); err != nil {
			w.logger.Infof("managed backend assist failed: %v", err)
		}
		backend = w.findAnyHealthyBackend()
	}

	if backend == nil {
		w.mu.Lock()
		w.failCount++
		fails := w.failCount
		w.mu.Unlock()
		w.logger.Infof("Desktop UP but backend still DOWN (fail=%d/%d)", fails, w.cfg.FailThreshold)
		if fails >= w.cfg.FailThreshold {
			restartPackagedDesktop(w.cfg, w.logger, w.back)
			w.mu.Lock()
			w.failCount = 0
			w.mu.Unlock()
			res := cycleResult{Desktop: "restarted", Backend: "respawning"}
			w.saveState(res)
			return res
		}
		res := cycleResult{Desktop: "up", Backend: "down"}
		w.saveState(res)
		return res
	}

	w.mu.Lock()
	w.failCount = 0
	w.mu.Unlock()
	w.logger.Infof("OK backend=pid:%d port:%d", backend.PID, backend.Port)
	res := cycleResult{
		Desktop:     "up",
		Backend:     "up",
		BackendPID:  backend.PID,
		BackendPort: backend.Port,
	}
	w.saveState(res)
	return res
}

func (w *Watchdog) RunLoop(stop <-chan struct{}) {
	w.logger.Infof("watchdog loop interval=%ds threshold=%d exe=%s", w.cfg.IntervalSec, w.cfg.FailThreshold, w.cfg.PackagedExe)
	for {
		w.RunCycle()
		if w.cfg.Once {
			return
		}
		select {
		case <-stop:
			return
		case <-time.After(time.Duration(w.cfg.IntervalSec) * time.Second):
		}
	}
}

type lockFile struct {
	PID       int    `json:"pid"`
	StartedAt string `json:"startedAt"`
	RepoRoot  string `json:"repoRoot"`
}

func acquireLock(lockPath, repoRoot string, logger *Logger) (func(), bool) {
	if fileExists(lockPath) {
		raw, err := os.ReadFile(lockPath)
		if err == nil {
			var lf lockFile
			if json.Unmarshal(raw, &lf) == nil && lf.PID > 0 {
				if processAlive(lf.PID) {
					logger.Infof("another watchdog holds %s (pid=%d) — exiting", lockPath, lf.PID)
					return nil, false
				}
			}
		}
		_ = os.Remove(lockPath)
	}
	lf := lockFile{
		PID:       os.Getpid(),
		StartedAt: time.Now().Format(time.RFC3339),
		RepoRoot:  repoRoot,
	}
	raw, _ := json.MarshalIndent(lf, "", "  ")
	if err := os.WriteFile(lockPath, raw, 0o644); err != nil {
		logger.Infof("failed to write lock: %v", err)
		return nil, false
	}
	release := func() {
		raw, err := os.ReadFile(lockPath)
		if err != nil {
			return
		}
		var existing lockFile
		if json.Unmarshal(raw, &existing) == nil && existing.PID == os.Getpid() {
			_ = os.Remove(lockPath)
		}
	}
	return release, true
}

func processAlive(pid int) bool {
	out, err := exec.Command("tasklist", "/FI", fmt.Sprintf("PID eq %d", pid), "/NH").Output()
	if err != nil {
		return false
	}
	return strings.Contains(string(out), fmt.Sprintf("%d", pid))
}
