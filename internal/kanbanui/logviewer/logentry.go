package logviewer

import "time"

// Level represents a log severity level.
type Level int

const (
	LevelDebug Level = iota
	LevelInfo
	LevelWarn
	LevelError
)

func (l Level) String() string {
	switch l {
	case LevelDebug:
		return "DEBUG"
	case LevelInfo:
		return "INFO"
	case LevelWarn:
		return "WARN"
	case LevelError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// Entry is a single log entry.
type Entry struct {
	Time    time.Time
	Level   Level
	Message string
}

// NewEntry creates a log entry with the current timestamp.
func NewEntry(level Level, message string) Entry {
	return Entry{
		Time:    time.Now(),
		Level:   level,
		Message: message,
	}
}
