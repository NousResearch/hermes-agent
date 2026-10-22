package logviewer

import (
	"testing"
	"time"
)

func TestLevelString(t *testing.T) {
	tests := []struct {
		level    Level
		expected string
	}{
		{LevelDebug, "DEBUG"},
		{LevelInfo, "INFO"},
		{LevelWarn, "WARN"},
		{LevelError, "ERROR"},
		{Level(99), "UNKNOWN"},
	}

	for _, tt := range tests {
		if got := tt.level.String(); got != tt.expected {
			t.Errorf("Level(%d).String() = %q, want %q", tt.level, got, tt.expected)
		}
	}
}

func TestNewEntry(t *testing.T) {
	entry := NewEntry(LevelInfo, "test message")

	if entry.Level != LevelInfo {
		t.Errorf("expected level Info, got %v", entry.Level)
	}
	if entry.Message != "test message" {
		t.Errorf("expected message 'test message', got %q", entry.Message)
	}
	if entry.Time.IsZero() {
		t.Error("expected non-zero time")
	}
	// Verify time is recent (within last second)
	if time.Since(entry.Time) > time.Second {
		t.Errorf("time is too old: %v", entry.Time)
	}
}
