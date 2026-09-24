package kanbanui

import (
	"fmt"
	"time"
)

// TaskStatus represents the lifecycle state of a kanban task.
// Constants are defined exactly as: triage, todo, ready, running, blocked, done.
type TaskStatus string

const (
	StatusTriage  TaskStatus = "triage"
	StatusTodo    TaskStatus = "todo"
	StatusReady   TaskStatus = "ready"
	StatusRunning TaskStatus = "running"
	StatusBlocked TaskStatus = "blocked"
	StatusDone    TaskStatus = "done"
)

// ValidStatuses returns the ordered list of all kanban status values.
// Order matches the typical left-to-right column layout on a board.
func ValidStatuses() []TaskStatus {
	return []TaskStatus{
		StatusTriage, StatusTodo, StatusReady,
		StatusRunning, StatusBlocked, StatusDone,
	}
}

// IsTerminal returns true if this status is a terminal state (done).
func (s TaskStatus) IsTerminal() bool {
	return s == StatusDone
}

// Task is the board-row representation of a kanban task.
type Task struct {
	ID         string
	Title      string
	Assignee   string
	Status     TaskStatus
	Priority   int
	CreatedAt  time.Time
	UpdatedAt  time.Time
}

// Comment represents a single comment on a task.
type Comment struct {
	ID        int
	TaskID    string
	Author    string
	Body      string
	CreatedAt time.Time
}

// TaskEvent represents a single event in a task's lifecycle.
type TaskEvent struct {
	TaskID    string
	Kind      string // e.g. "created", "claimed", "completed", "crashed", "heartbeat"
	Payload   string // JSON payload or human-readable summary
	CreatedAt time.Time
	RunID     int
}

// Message returns a human-readable log line for this event.
func (e TaskEvent) Message() string {
	ts := e.CreatedAt.Format("15:04:05")
	return fmt.Sprintf("[%s] %s (run #%d)", ts, e.Kind, e.RunID)
}

// TaskDetail is the full task representation including comments and parent/child relationships.
type TaskDetail struct {
	Task
	Body       string
	CreatedBy  string
	Comments   []Comment
	ParentIDs  []string
	ChildIDs   []string
}

// Column groups tasks under a single status for board rendering.
type Column struct {
	Status TaskStatus
	Tasks  []Task
}

// BoardSnapshot is the in-memory representation the renderer consumes.
// Tasks are pre-grouped into columns by status, sorted by assignee then title within each column.
type BoardSnapshot struct {
	Columns  []Column
	LoadedAt time.Time
	Source   string
	LoadError error
}

// AllTasks flattens all columns into a single slice (preserving column order).
func (bs BoardSnapshot) AllTasks() []Task {
	total := 0
	for _, col := range bs.Columns {
		total += len(col.Tasks)
	}
	if total == 0 {
		return nil
	}
	all := make([]Task, 0, total)
	for _, col := range bs.Columns {
		all = append(all, col.Tasks...)
	}
	return all
}

// PlaceholderTasks returns deterministic rows until the SQLite store lands.
func PlaceholderTasks() []Task {
	now := time.Now()
	return []Task{
		{ID: "t_scaffold", Title: "Scaffold Go module and Bubble Tea entry point", Assignee: "architect", Status: StatusRunning, Priority: 1, UpdatedAt: now},
		{ID: "t_data", Title: "Connect kanban TUI to SQLite board data", Assignee: "coder", Status: StatusTodo, Priority: 0, UpdatedAt: now},
		{ID: "t_views", Title: "Implement board columns and detail pane", Assignee: "coder", Status: StatusTodo, Priority: 0, UpdatedAt: now},
	}
}
