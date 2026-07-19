package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
)

type HTTPServer struct {
	cfg      Config
	wd       *Watchdog
	shutdown func()
	mu       sync.Mutex
}

func NewHTTPServer(cfg Config, wd *Watchdog, shutdown func()) *HTTPServer {
	return &HTTPServer{cfg: cfg, wd: wd, shutdown: shutdown}
}

func (s *HTTPServer) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/api/status", s.handleStatus)
	mux.HandleFunc("/api/v1/status", s.handleStatus)
	mux.HandleFunc("/api/v1/pause", s.handlePause)
	mux.HandleFunc("/api/v1/resume", s.handleResume)
	mux.HandleFunc("/api/v1/cycle", s.handleCycle)
	mux.HandleFunc("/api/v1/stop", s.handleStop)
	return mux
}

func (s *HTTPServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *HTTPServer) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, s.wd.State())
}

func (s *HTTPServer) handlePause(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdmin(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.wd.SetPaused(true)
	writeJSON(w, http.StatusOK, map[string]any{"paused": true})
}

func (s *HTTPServer) handleResume(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdmin(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	s.wd.SetPaused(false)
	writeJSON(w, http.StatusOK, map[string]any{"paused": false})
}

func (s *HTTPServer) handleCycle(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdmin(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	result := s.wd.RunCycle()
	writeJSON(w, http.StatusOK, result)
}

func (s *HTTPServer) handleStop(w http.ResponseWriter, r *http.Request) {
	if !s.requireAdmin(w, r) {
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"stopping": "true"})
	go s.shutdown()
}

func (s *HTTPServer) requireAdmin(w http.ResponseWriter, r *http.Request) bool {
	token := strings.TrimSpace(s.cfg.AdminToken)
	if token == "" {
		http.Error(w, "admin token not configured — mutating API disabled", http.StatusForbidden)
		return false
	}
	got := extractAdminToken(r)
	if got == "" || got != token {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return false
	}
	return true
}

func extractAdminToken(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	if strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		return strings.TrimSpace(auth[7:])
	}
	if v := strings.TrimSpace(r.Header.Get("X-Admin-Token")); v != "" {
		return v
	}
	return strings.TrimSpace(r.URL.Query().Get("admin_token"))
}

func writeJSON(w http.ResponseWriter, code int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(payload)
}
