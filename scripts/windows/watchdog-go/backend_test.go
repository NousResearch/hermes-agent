package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseReadyPortLine(t *testing.T) {
	cases := []struct {
		line string
		want int
		ok   bool
	}{
		{"HERMES_BACKEND_READY port=43210", 43210, true},
		{"HERMES_DASHBOARD_READY port=9123", 9123, true},
		{"noise", 0, false},
		{"HERMES_BACKEND_READY port=0", 0, false},
	}
	for _, tc := range cases {
		got, ok := parseReadyPortLine(tc.line)
		if ok != tc.ok || got != tc.want {
			t.Fatalf("line %q => (%d,%v) want (%d,%v)", tc.line, got, ok, tc.want, tc.ok)
		}
	}
}

func TestResolvePythonExe(t *testing.T) {
	dir := t.TempDir()
	venvPy := filepath.Join(dir, ".venv", "Scripts")
	if err := os.MkdirAll(venvPy, 0o755); err != nil {
		t.Fatal(err)
	}
	pyPath := filepath.Join(venvPy, "python.exe")
	if err := os.WriteFile(pyPath, []byte("stub"), 0o644); err != nil {
		t.Fatal(err)
	}
	got := resolvePythonExe(dir)
	if got != pyPath {
		t.Fatalf("expected %q got %q", pyPath, got)
	}
}

func TestDesktopLaunchEnvIncludesRemoteWhenManifest(t *testing.T) {
	cfg := Config{
		HermesRoot: `C:\repo`,
		HermesHome: `C:\Users\u\.hermes`,
	}
	manifest := &DesktopBackendManifest{
		BaseURL: "http://127.0.0.1:54321",
		Token:   "tok",
	}
	env := desktopLaunchEnv(cfg, manifest)
	joined := stringsJoinEnv(env)
	for _, want := range []string{
		"HERMES_DESKTOP_REMOTE_URL=http://127.0.0.1:54321",
		"HERMES_DESKTOP_REMOTE_TOKEN=tok",
		"HERMES_DESKTOP_HERMES_ROOT=C:\\repo",
	} {
		if !containsSubstr(joined, want) {
			t.Fatalf("missing %q in %q", want, joined)
		}
	}
}

func stringsJoinEnv(env []string) string {
	out := ""
	for _, e := range env {
		out += e + ";"
	}
	return out
}

func containsSubstr(haystack, needle string) bool {
	return len(needle) == 0 || (len(haystack) >= len(needle) && indexOf(haystack, needle) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

func TestBackendManagerWriteReadManifest(t *testing.T) {
	dir := t.TempDir()
	cfg := Config{
		DataDir:    dir,
		HermesRoot: dir,
		HermesHome: dir,
	}
	logger := NewLogger(filepath.Join(dir, "test.log"))
	bm := NewBackendManager(cfg, logger)
	bm.mu.Lock()
	bm.token = "abc"
	bm.mu.Unlock()
	if err := bm.publishManifestLocked(12345, 999); err != nil {
		t.Fatal(err)
	}
	got, err := bm.readManifest()
	if err != nil {
		t.Fatal(err)
	}
	if got.Port != 12345 || got.Token != "abc" || !got.Managed {
		t.Fatalf("unexpected manifest: %+v", got)
	}
}
