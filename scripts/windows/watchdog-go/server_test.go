package main

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
)

func TestRequireAdminRejectsEmptyToken(t *testing.T) {
	cfg := Config{AdminToken: ""}
	wd := NewWatchdog(cfg, NewLogger(t.TempDir() + "/test.log"))
	srv := NewHTTPServer(cfg, wd, func() {})
	req := httptest.NewRequest("POST", "/api/v1/pause", nil)
	req.Header.Set("Authorization", "Bearer anything")
	w := httptest.NewRecorder()
	srv.handlePause(w, req)
	if w.Code != 403 {
		t.Fatalf("expected 403, got %d", w.Code)
	}
}

func TestRequireAdminRejectsWrongToken(t *testing.T) {
	cfg := Config{AdminToken: "secret-token"}
	wd := NewWatchdog(cfg, NewLogger(t.TempDir()+"/test.log"))
	srv := NewHTTPServer(cfg, wd, func() {})
	req := httptest.NewRequest("POST", "/api/v1/pause", nil)
	req.Header.Set("X-Admin-Token", "wrong")
	w := httptest.NewRecorder()
	srv.handlePause(w, req)
	if w.Code != 401 {
		t.Fatalf("expected 401, got %d", w.Code)
	}
}

func TestRequireAdminAcceptsBearer(t *testing.T) {
	cfg := Config{AdminToken: "secret-token"}
	wd := NewWatchdog(cfg, NewLogger(t.TempDir()+"/test.log"))
	srv := NewHTTPServer(cfg, wd, func() {})
	req := httptest.NewRequest("POST", "/api/v1/pause", nil)
	req.Header.Set("Authorization", "Bearer secret-token")
	w := httptest.NewRecorder()
	srv.handlePause(w, req)
	if w.Code != 200 {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	if !wd.IsPaused() {
		t.Fatal("expected paused")
	}
}

func TestStatusJSON(t *testing.T) {
	cfg := Config{AdminToken: "x", ListenAddr: "127.0.0.1:9920"}
	wd := NewWatchdog(cfg, NewLogger(t.TempDir()+"/test.log"))
	srv := NewHTTPServer(cfg, wd, func() {})
	req := httptest.NewRequest("GET", "/api/status", nil)
	w := httptest.NewRecorder()
	srv.handleStatus(w, req)
	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var state WatchdogState
	if err := json.Unmarshal(w.Body.Bytes(), &state); err != nil {
		t.Fatal(err)
	}
	if state.ListenAddr != cfg.ListenAddr {
		t.Fatalf("unexpected listen addr %q", state.ListenAddr)
	}
}

func TestIsDesktopBackendCommandLine(t *testing.T) {
	cases := []struct {
		cl   string
		want bool
	}{
		{"python -m hermes_cli.main serve", true},
		{"python -m hermes_cli.main serve --host 127.0.0.1 --port 0", true},
		{"python -m hermes_cli.main serve --port 9120", false},
		{"python -m hermes_cli.main dashboard --no-open", true},
		{"python -m hermes_cli.main gateway start", false},
		{"python -m hermes_cli.main harness start", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := isDesktopBackendCommandLine(tc.cl); got != tc.want {
			t.Fatalf("cmd %q => %v want %v", tc.cl, got, tc.want)
		}
	}
}

func TestIsReservedOpsPort(t *testing.T) {
	if !isReservedOpsPort(9120) || !isReservedOpsPort(8787) {
		t.Fatal("expected 9120/8787 reserved")
	}
	if isReservedOpsPort(54321) {
		t.Fatal("ephemeral port must not be reserved")
	}
}
