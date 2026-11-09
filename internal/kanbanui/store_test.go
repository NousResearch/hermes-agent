package kanbanui

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// --- Helpers ---

// newInMemoryStore creates a Store backed by an in-memory SQLite database
// with the tasks table pre-created. Returns the Store and a helper to insert tasks.
func newInMemoryStore(t *testing.T) Store {
	t.Helper()
	store := NewStore(":memory:")
	if store.db == nil {
		t.Fatal("failed to create in-memory store")
	}
	return store
}

// setupTestSchema creates the tasks table in the given store's database.
func setupTestSchema(t *testing.T, s Store) {
	t.Helper()
	if s.db == nil {
		t.Fatal("store has no database")
	}
	_, err := s.db.Exec(`
		CREATE TABLE tasks (
			id TEXT PRIMARY KEY,
			title TEXT NOT NULL,
			body TEXT,
			assignee TEXT,
			status TEXT NOT NULL,
			priority INTEGER DEFAULT 0,
			created_by TEXT,
			created_at INTEGER NOT NULL,
			started_at INTEGER,
			completed_at INTEGER,
			last_heartbeat_at INTEGER
		)
	`)
	if err != nil {
		t.Fatalf("create tasks table: %v", err)
	}
}

// insertTask inserts a single task row into the store.
func insertTask(t *testing.T, s Store, id, title, assignee, status string, priority int) {
	t.Helper()
	if s.db == nil {
		t.Fatal("store has no database")
	}
	now := time.Now().Unix()
	_, err := s.db.Exec(
		"INSERT INTO tasks (id, title, assignee, status, priority, created_at) VALUES (?, ?, ?, ?, ?, ?)",
		id, title, assignee, status, priority, now,
	)
	if err != nil {
		t.Fatalf("insert task %s: %v", id, err)
	}
}

// setupCommentsTable creates the task_comments table.
func setupCommentsTable(t *testing.T, s Store) {
	t.Helper()
	if s.db == nil {
		t.Fatal("store has no database")
	}
	_, err := s.db.Exec(`
		CREATE TABLE task_comments (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			task_id TEXT NOT NULL,
			author TEXT NOT NULL,
			body TEXT NOT NULL,
			created_at INTEGER NOT NULL
		)
	`)
	if err != nil {
		t.Fatalf("create comments table: %v", err)
	}
}

// setupLinksTable creates the task_links table.
func setupLinksTable(t *testing.T, s Store) {
	t.Helper()
	if s.db == nil {
		t.Fatal("store has no database")
	}
	_, err := s.db.Exec(`
		CREATE TABLE task_links (
			parent_id TEXT NOT NULL,
			child_id  TEXT NOT NULL
		)
	`)
	if err != nil {
		t.Fatalf("create links table: %v", err)
	}
}

// --- Status Constants Tests ---

func TestTaskStatusConstants(t *testing.T) {
	statuses := ValidStatuses()
	expected := []TaskStatus{
		StatusTriage, StatusTodo, StatusReady,
		StatusRunning, StatusBlocked, StatusDone,
	}

	if len(statuses) != len(expected) {
		t.Fatalf("ValidStatuses() = %d, want %d", len(statuses), len(expected))
	}

	for i, want := range expected {
		if statuses[i] != want {
			t.Errorf("status[%d] = %q, want %q", i, statuses[i], want)
		}
	}
}

func TestTaskStatusValues(t *testing.T) {
	// Verify the string values match the spec exactly.
	tests := []struct {
		status TaskStatus
		want   string
	}{
		{StatusTriage, "triage"},
		{StatusTodo, "todo"},
		{StatusReady, "ready"},
		{StatusRunning, "running"},
		{StatusBlocked, "blocked"},
		{StatusDone, "done"},
	}
	for _, tt := range tests {
		if string(tt.status) != tt.want {
			t.Errorf("TaskStatus(%q) = %q, want %q", tt.status, string(tt.status), tt.want)
		}
	}
}

func TestTaskStatusIsTerminal(t *testing.T) {
	if !StatusDone.IsTerminal() {
		t.Error("StatusDone.IsTerminal() = false, want true")
	}
	for _, s := range []TaskStatus{StatusTriage, StatusTodo, StatusReady, StatusRunning, StatusBlocked} {
		if s.IsTerminal() {
			t.Errorf("%q.IsTerminal() = true, want false", s)
		}
	}
}

// --- Store Construction Tests ---

func TestNewStoreInMemory(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()

	if store.Path() != ":memory:" {
		t.Errorf("Path() = %q, want %q", store.Path(), ":memory:")
	}
}

func TestNewStoreInvalidPath(t *testing.T) {
	store := NewStore("/nonexistent/path/kanban.db")
	if store.db != nil {
		_ = store.db.Close()
	}
	// Should return a store with nil db when path doesn't exist.
	snapshot := store.Load()
	if snapshot.LoadError == nil {
		t.Fatal("expected error for non-existent DB, got nil")
	}
}

func TestStorePath(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	if store.Path() != ":memory:" {
		t.Errorf("Path() = %q, want %q", store.Path(), ":memory:")
	}
}

func TestStoreCloseNil(t *testing.T) {
	// Closing a store with nil db must not panic.
	store := Store{dbPath: "nope"}
	err := store.Close()
	if err != nil {
		t.Errorf("Close() on nil db = %v, want nil", err)
	}
}

// --- Task Loading Tests ---

func TestLoadEmptyDatabase(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	// Should have all 6 columns (even if empty).
	if len(snapshot.Columns) != len(ValidStatuses()) {
		t.Errorf("columns = %d, want %d", len(snapshot.Columns), len(ValidStatuses()))
	}

	// No tasks at all.
	if len(snapshot.AllTasks()) != 0 {
		t.Errorf("AllTasks() = %d, want 0", len(snapshot.AllTasks()))
	}
}

func TestLoadSingleTask(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)
	insertTask(t, store, "t1", "First Task", "coder", "todo", 0)

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	all := snapshot.AllTasks()
	if len(all) != 1 {
		t.Fatalf("AllTasks() = %d, want 1", len(all))
	}
	if all[0].ID != "t1" {
		t.Errorf("task ID = %q, want %q", all[0].ID, "t1")
	}
	if all[0].Title != "First Task" {
		t.Errorf("task Title = %q, want %q", all[0].Title, "First Task")
	}
	if all[0].Assignee != "coder" {
		t.Errorf("task Assignee = %q, want %q", all[0].Assignee, "coder")
	}
	if all[0].Status != StatusTodo {
		t.Errorf("task Status = %q, want %q", all[0].Status, StatusTodo)
	}
}

func TestLoadMultipleTasks(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	insertTask(t, store, "t1", "Alpha", "bob", "todo", 0)
	insertTask(t, store, "t2", "Beta", "alice", "todo", 1)
	insertTask(t, store, "t3", "Gamma", "alice", "running", 0)
	insertTask(t, store, "t4", "Delta", "bob", "done", 0)

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	all := snapshot.AllTasks()
	if len(all) != 4 {
		t.Fatalf("AllTasks() = %d, want 4", len(all))
	}
}

func TestLoadTaskWithNullAssignee(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	// Insert with NULL assignee.
	if store.db == nil {
		t.Fatal("no db")
	}
	_, err := store.db.Exec(
		"INSERT INTO tasks (id, title, assignee, status, priority, created_at) VALUES (?, ?, NULL, ?, ?, ?)",
		"t1", "Unassigned Task", "todo", 0, time.Now().Unix(),
	)
	if err != nil {
		t.Fatalf("insert: %v", err)
	}

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	all := snapshot.AllTasks()
	if len(all) != 1 {
		t.Fatalf("AllTasks() = %d, want 1", len(all))
	}
	if all[0].Assignee != "" {
		t.Errorf("Assignee = %q, want empty string for NULL", all[0].Assignee)
	}
}

func TestLoadSnapshotMetadata(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	if snapshot.Source != ":memory:" {
		t.Errorf("Source = %q, want %q", snapshot.Source, ":memory:")
	}
	if snapshot.LoadedAt.IsZero() {
		t.Error("LoadedAt is zero")
	}
}

// --- Grouping into Statuses Tests ---

func TestColumnsFromTasksEmpty(t *testing.T) {
	columns := columnsFromTasks(nil)
	// Should still produce all 6 columns (empty).
	if len(columns) != len(ValidStatuses()) {
		t.Errorf("columns = %d, want %d", len(columns), len(ValidStatuses()))
	}
	for _, col := range columns {
		if len(col.Tasks) != 0 {
			t.Errorf("column %q has %d tasks, want 0", col.Status, len(col.Tasks))
		}
	}
}

func TestColumnsFromTasksAllStatuses(t *testing.T) {
	tasks := []Task{
		{ID: "t1", Title: "Triage Task", Status: StatusTriage},
		{ID: "t2", Title: "Todo Task", Status: StatusTodo},
		{ID: "t3", Title: "Ready Task", Status: StatusReady},
		{ID: "t4", Title: "Running Task", Status: StatusRunning},
		{ID: "t5", Title: "Blocked Task", Status: StatusBlocked},
		{ID: "t6", Title: "Done Task", Status: StatusDone},
	}

	columns := columnsFromTasks(tasks)
	if len(columns) != 6 {
		t.Fatalf("columns = %d, want 6", len(columns))
	}

	// Verify each column has exactly one task with the correct status.
	for i, col := range columns {
		if col.Status != ValidStatuses()[i] {
			t.Errorf("column[%d].Status = %q, want %q", i, col.Status, ValidStatuses()[i])
		}
		if len(col.Tasks) != 1 {
			t.Errorf("column[%d] has %d tasks, want 1", i, len(col.Tasks))
		}
	}
}

func TestColumnsFromTasksMultiplePerStatus(t *testing.T) {
	tasks := []Task{
		{ID: "t1", Title: "A", Status: StatusTodo},
		{ID: "t2", Title: "B", Status: StatusTodo},
		{ID: "t3", Title: "C", Status: StatusTodo},
		{ID: "t4", Title: "D", Status: StatusDone},
		{ID: "t5", Title: "E", Status: StatusDone},
	}

	columns := columnsFromTasks(tasks)

	// Find todo column.
	var todoCol, doneCol *Column
	for i := range columns {
		if columns[i].Status == StatusTodo {
			todoCol = &columns[i]
		}
		if columns[i].Status == StatusDone {
			doneCol = &columns[i]
		}
	}
	if todoCol == nil || doneCol == nil {
		t.Fatal("missing todo or done column")
	}

	if len(todoCol.Tasks) != 3 {
		t.Errorf("todo column = %d tasks, want 3", len(todoCol.Tasks))
	}
	if len(doneCol.Tasks) != 2 {
		t.Errorf("done column = %d tasks, want 2", len(doneCol.Tasks))
	}
}

func TestColumnsFromTasksEmptyColumnsIncluded(t *testing.T) {
	// Only insert tasks in one status; all others should still appear.
	tasks := []Task{
		{ID: "t1", Title: "Only Task", Status: StatusDone},
	}

	columns := columnsFromTasks(tasks)

	// All 6 columns must exist.
	if len(columns) != 6 {
		t.Fatalf("columns = %d, want 6", len(columns))
	}

	// Empty columns should have nil/empty task slices.
	for _, col := range columns {
		if col.Status != StatusDone && len(col.Tasks) != 0 {
			t.Errorf("column %q should be empty but has %d tasks", col.Status, len(col.Tasks))
		}
	}
}

func TestColumnsFromTasksCanonicalOrder(t *testing.T) {
	// Insert tasks in reverse order; columns must still be canonical.
	tasks := []Task{
		{ID: "t1", Title: "X", Status: StatusDone},
		{ID: "t2", Title: "Y", Status: StatusRunning},
		{ID: "t3", Title: "Z", Status: StatusTriage},
	}

	columns := columnsFromTasks(tasks)

	expected := []TaskStatus{StatusTriage, StatusTodo, StatusReady, StatusRunning, StatusBlocked, StatusDone}
	for i, col := range columns {
		if col.Status != expected[i] {
			t.Errorf("column[%d].Status = %q, want %q", i, col.Status, expected[i])
		}
	}
}

func TestLoadGroupsByStatus(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	// Insert tasks across different statuses.
	insertTask(t, store, "t1", "A", "coder", "triage", 0)
	insertTask(t, store, "t2", "B", "coder", "todo", 0)
	insertTask(t, store, "t3", "C", "coder", "running", 0)
	insertTask(t, store, "t4", "D", "coder", "done", 0)

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	// Verify each task is in the correct column.
	for _, col := range snapshot.Columns {
		for _, task := range col.Tasks {
			if task.Status != col.Status {
				t.Errorf("task %q (status=%q) in column %q", task.ID, task.Status, col.Status)
			}
		}
	}
}

// --- Sorting Tests ---

func TestSortTasksByAssigneeTitle(t *testing.T) {
	tasks := []Task{
		{ID: "t1", Title: "Zebra", Assignee: "bob", Status: StatusTodo},
		{ID: "t2", Title: "Alpha", Assignee: "alice", Status: StatusTodo},
		{ID: "t3", Title: "Beta", Assignee: "alice", Status: StatusTodo},
		{ID: "t4", Title: "Alpha", Assignee: "bob", Status: StatusTodo},
		{ID: "t5", Title: "Charlie", Assignee: "charlie", Status: StatusTodo},
	}

	sortTasksByAssigneeTitle(tasks)

	// Expected: alice/Alpha, alice/Beta, bob/Alpha, bob/Zebra, charlie/Charlie
	expected := []struct {
		assignee string
		title    string
	}{
		{"alice", "Alpha"},
		{"alice", "Beta"},
		{"bob", "Alpha"},
		{"bob", "Zebra"},
		{"charlie", "Charlie"},
	}

	for i, want := range expected {
		if tasks[i].Assignee != want.assignee {
			t.Errorf("tasks[%d].Assignee = %q, want %q", i, tasks[i].Assignee, want.assignee)
		}
		if tasks[i].Title != want.title {
			t.Errorf("tasks[%d].Title = %q, want %q", i, tasks[i].Title, want.title)
		}
	}
}

func TestSortTasksByAssigneeTitleEmpty(t *testing.T) {
	// Must not panic on empty or nil slices.
	sortTasksByAssigneeTitle(nil)
	sortTasksByAssigneeTitle([]Task{})
}

func TestSortTasksByAssigneeTitleSingle(t *testing.T) {
	tasks := []Task{{ID: "t1", Title: "Only", Assignee: "solo", Status: StatusTodo}}
	sortTasksByAssigneeTitle(tasks)
	if tasks[0].ID != "t1" {
		t.Errorf("single task moved: ID = %q", tasks[0].ID)
	}
}

func TestSortTasksByAssigneeTitleWithEmptyAssignee(t *testing.T) {
	tasks := []Task{
		{ID: "t1", Title: "Z", Assignee: "bob", Status: StatusTodo},
		{ID: "t2", Title: "A", Assignee: "", Status: StatusTodo},
		{ID: "t3", Title: "M", Assignee: "alice", Status: StatusTodo},
	}

	sortTasksByAssigneeTitle(tasks)

	// Empty assignee sorts first (empty string < "alice" < "bob").
	if tasks[0].Assignee != "" {
		t.Errorf("first task Assignee = %q, want empty", tasks[0].Assignee)
	}
	if tasks[1].Assignee != "alice" {
		t.Errorf("second task Assignee = %q, want alice", tasks[1].Assignee)
	}
	if tasks[2].Assignee != "bob" {
		t.Errorf("third task Assignee = %q, want bob", tasks[2].Assignee)
	}
}

func TestLoadSortedWithinColumns(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	// Insert tasks in reverse order within the same status.
	insertTask(t, store, "t1", "Zebra", "bob", "todo", 0)
	insertTask(t, store, "t2", "Alpha", "alice", "todo", 0)
	insertTask(t, store, "t3", "Beta", "alice", "todo", 0)
	insertTask(t, store, "t4", "Alpha", "bob", "todo", 0)

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	// Find the todo column.
	var todoCol *Column
	for i := range snapshot.Columns {
		if snapshot.Columns[i].Status == StatusTodo {
			todoCol = &snapshot.Columns[i]
			break
		}
	}
	if todoCol == nil || len(todoCol.Tasks) != 4 {
		t.Fatalf("todo column: got %v tasks", todoCol)
	}

	// Verify sort order: alice/Alpha, alice/Beta, bob/Alpha, bob/Zebra
	expected := []struct {
		assignee string
		title    string
	}{
		{"alice", "Alpha"},
		{"alice", "Beta"},
		{"bob", "Alpha"},
		{"bob", "Zebra"},
	}

	for i, want := range expected {
		task := todoCol.Tasks[i]
		if task.Assignee != want.assignee || task.Title != want.title {
			t.Errorf("todo[%d] = (%q/%q), want (%q/%q)", i, task.Assignee, task.Title, want.assignee, want.title)
		}
	}
}

// --- UpdateTaskStatus Tests ---

func TestUpdateTaskStatusValid(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)
	insertTask(t, store, "t1", "Task", "coder", "todo", 0)

	err := store.UpdateTaskStatus("t1", StatusRunning)
	if err != nil {
		t.Fatalf("UpdateTaskStatus() = %v, want nil", err)
	}

	// Verify the change persisted.
	var status string
	err = store.db.QueryRow("SELECT status FROM tasks WHERE id = ?", "t1").Scan(&status)
	if err != nil {
		t.Fatalf("verify query: %v", err)
	}
	if status != "running" {
		t.Errorf("status = %q, want %q", status, "running")
	}
}

func TestUpdateTaskStatusInvalid(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)
	insertTask(t, store, "t1", "Task", "coder", "todo", 0)

	err := store.UpdateTaskStatus("t1", TaskStatus("invalid_status"))
	if err == nil {
		t.Fatal("UpdateTaskStatus(invalid) = nil, want error")
	}
}

func TestUpdateTaskStatusNotFound(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	err := store.UpdateTaskStatus("nonexistent", StatusDone)
	if err == nil {
		t.Fatal("UpdateTaskStatus(nonexistent) = nil, want error")
	}
}

func TestUpdateTaskStatusNilDB(t *testing.T) {
	store := Store{dbPath: "nope"}
	err := store.UpdateTaskStatus("t1", StatusDone)
	if err == nil {
		t.Fatal("UpdateTaskStatus on nil db = nil, want error")
	}
}

// --- UpdateTaskAssignee Tests ---

func TestUpdateTaskAssignee(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)
	insertTask(t, store, "t1", "Task", "coder", "todo", 0)

	err := store.UpdateTaskAssignee("t1", "reviewer")
	if err != nil {
		t.Fatalf("UpdateTaskAssignee() = %v, want nil", err)
	}

	var assignee string
	err = store.db.QueryRow("SELECT assignee FROM tasks WHERE id = ?", "t1").Scan(&assignee)
	if err != nil {
		t.Fatalf("verify query: %v", err)
	}
	if assignee != "reviewer" {
		t.Errorf("assignee = %q, want %q", assignee, "reviewer")
	}
}

func TestUpdateTaskAssigneeNotFound(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	err := store.UpdateTaskAssignee("nonexistent", "coder")
	if err == nil {
		t.Fatal("UpdateTaskAssignee(nonexistent) = nil, want error")
	}
}

func TestUpdateTaskAssigneeNilDB(t *testing.T) {
	store := Store{dbPath: "nope"}
	err := store.UpdateTaskAssignee("t1", "coder")
	if err == nil {
		t.Fatal("UpdateTaskAssignee on nil db = nil, want error")
	}
}

// --- GetTaskDetail Tests ---

func TestGetTaskDetail(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)
	setupCommentsTable(t, store)
	setupLinksTable(t, store)
	insertTask(t, store, "t1", "Parent Task", "coder", "running", 1)
	insertTask(t, store, "t2", "Child Task", "coder", "todo", 0)

	// Insert comment.
	if store.db == nil {
		t.Fatal("no db")
	}
	_, err := store.db.Exec(
		"INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?, ?, ?, ?)",
		"t1", "reviewer", "LGTM", time.Now().Unix(),
	)
	if err != nil {
		t.Fatalf("insert comment: %v", err)
	}

	// Insert parent-child link.
	_, err = store.db.Exec(
		"INSERT INTO task_links (parent_id, child_id) VALUES (?, ?)",
		"t1", "t2",
	)
	if err != nil {
		t.Fatalf("insert link: %v", err)
	}

	detail, err := store.GetTaskDetail("t1")
	if err != nil {
		t.Fatalf("GetTaskDetail() = _, %v, want _, nil", err)
	}

	if detail.ID != "t1" {
		t.Errorf("ID = %q, want %q", detail.ID, "t1")
	}
	if detail.Title != "Parent Task" {
		t.Errorf("Title = %q, want %q", detail.Title, "Parent Task")
	}
	if detail.Status != StatusRunning {
		t.Errorf("Status = %q, want %q", detail.Status, StatusRunning)
	}
	if detail.Priority != 1 {
		t.Errorf("Priority = %d, want 1", detail.Priority)
	}
	if len(detail.Comments) != 1 {
		t.Errorf("Comments = %d, want 1", len(detail.Comments))
	}
	if len(detail.ChildIDs) != 1 || detail.ChildIDs[0] != "t2" {
		t.Errorf("ChildIDs = %v, want [t2]", detail.ChildIDs)
	}
}

func TestGetTaskDetailNotFound(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)
	setupCommentsTable(t, store)
	setupLinksTable(t, store)

	_, err := store.GetTaskDetail("nonexistent")
	if err == nil {
		t.Fatal("GetTaskDetail(nonexistent) = _, nil, want error")
	}
}

func TestGetTaskDetailNilDB(t *testing.T) {
	store := Store{dbPath: "nope"}
	_, err := store.GetTaskDetail("t1")
	if err == nil {
		t.Fatal("GetTaskDetail on nil db = _, nil, want error")
	}
}

// --- BoardSnapshot Tests ---

func TestBoardSnapshotAllTasks(t *testing.T) {
	snapshot := BoardSnapshot{
		Columns: []Column{
			{Status: StatusTodo, Tasks: []Task{
				{ID: "a", Title: "A", Status: StatusTodo},
				{ID: "b", Title: "B", Status: StatusTodo},
			}},
			{Status: StatusDone, Tasks: []Task{
				{ID: "c", Title: "C", Status: StatusDone},
			}},
		},
	}

	all := snapshot.AllTasks()
	if len(all) != 3 {
		t.Errorf("AllTasks() = %d, want 3", len(all))
	}

	// Verify column order is preserved.
	if all[0].ID != "a" || all[1].ID != "b" || all[2].ID != "c" {
		t.Errorf("AllTasks() order = %v, want [a b c]", []string{all[0].ID, all[1].ID, all[2].ID})
	}
}

func TestBoardSnapshotAllTasksEmpty(t *testing.T) {
	snapshot := BoardSnapshot{Columns: []Column{}}
	all := snapshot.AllTasks()
	if all != nil {
		t.Errorf("AllTasks() on empty snapshot = %v, want nil", all)
	}
}

func TestBoardSnapshotAllTasksEmptyColumns(t *testing.T) {
	// Columns exist but have no tasks.
	snapshot := BoardSnapshot{
		Columns: []Column{
			{Status: StatusTodo, Tasks: nil},
			{Status: StatusDone, Tasks: []Task{}},
		},
	}
	all := snapshot.AllTasks()
	if all != nil {
		t.Errorf("AllTasks() on empty columns = %v, want nil", all)
	}
}

// --- PlaceholderTasks Tests ---

func TestPlaceholderTasks(t *testing.T) {
	tasks := PlaceholderTasks()
	if len(tasks) == 0 {
		t.Fatal("PlaceholderTasks() returned empty slice")
	}

	for _, task := range tasks {
		if task.ID == "" {
			t.Error("placeholder task has empty ID")
		}
		if task.Title == "" {
			t.Error("placeholder task has empty Title")
		}
		if task.Status == "" {
			t.Error("placeholder task has empty Status")
		}
		if task.UpdatedAt.IsZero() {
			t.Error("placeholder task has zero UpdatedAt")
		}
	}
}

func TestPlaceholderTasksInColumns(t *testing.T) {
	// Placeholder tasks should group correctly into columns.
	tasks := PlaceholderTasks()
	columns := columnsFromTasks(tasks)

	if len(columns) != len(ValidStatuses()) {
		t.Errorf("columns from placeholders = %d, want %d", len(columns), len(ValidStatuses()))
	}

	// Count total tasks across columns.
	total := 0
	for _, col := range columns {
		total += len(col.Tasks)
	}
	if total != len(tasks) {
		t.Errorf("total tasks in columns = %d, want %d", total, len(tasks))
	}
}

// --- Load with nil DB (fallback) Tests ---

func TestLoadNilDB(t *testing.T) {
	store := Store{dbPath: "fallback.db"}
	snapshot := store.Load()

	if snapshot.LoadError == nil {
		t.Fatal("expected error for nil db, got nil")
	}

	// Should still have placeholder columns.
	if len(snapshot.Columns) == 0 {
		t.Fatal("expected fallback columns, got none")
	}

	// Should have placeholder tasks.
	all := snapshot.AllTasks()
	if len(all) == 0 {
		t.Error("expected fallback tasks, got none")
	}
}

func TestLoadNilDBSource(t *testing.T) {
	store := Store{dbPath: "fallback.db"}
	snapshot := store.Load()

	if snapshot.Source != "fallback.db" {
		t.Errorf("Source = %q, want %q", snapshot.Source, "fallback.db")
	}
	if snapshot.LoadedAt.IsZero() {
		t.Error("LoadedAt is zero even on error")
	}
}

// --- Integration test against real kanban.db ---

func TestStoreLoad(t *testing.T) {
	// Use the real kanban database if it exists.
	dbPath := os.Getenv("HERMES_KANBAN_DB")
	if dbPath == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			t.Skip("no home dir")
		}
		dbPath = filepath.Join(home, ".hermes", "kanban.db")
	}

	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		t.Skip("kanban.db not found at", dbPath)
	}

	store := NewStore(dbPath)
	defer store.Close()

	if store.Path() != dbPath {
		t.Errorf("Path() = %q, want %q", store.Path(), dbPath)
	}

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	if len(snapshot.Columns) == 0 {
		t.Fatal("Load() returned no columns")
	}

	// Verify canonical column order.
	expected := ValidStatuses()
	if len(snapshot.Columns) != len(expected) {
		t.Errorf("columns = %d, want %d", len(snapshot.Columns), len(expected))
	}
	for i, col := range snapshot.Columns {
		if i < len(expected) && col.Status != expected[i] {
			t.Errorf("column[%d].Status = %q, want %q", i, col.Status, expected[i])
		}
	}

	// Verify tasks are sorted within columns.
	for _, col := range snapshot.Columns {
		for j := 1; j < len(col.Tasks); j++ {
			prev := col.Tasks[j-1]
			curr := col.Tasks[j]
			if prev.Assignee > curr.Assignee ||
				(prev.Assignee == curr.Assignee && prev.Title > curr.Title) {
				t.Errorf("tasks not sorted in column %q: %q > %q", col.Status, prev.Title, curr.Title)
			}
		}
	}

	t.Logf("Loaded %d columns, %d total tasks from %s", len(snapshot.Columns), len(snapshot.AllTasks()), snapshot.Source)
}

func TestStoreLoadFallback(t *testing.T) {
	// Test with a non-existent database - should fall back to placeholders.
	store := NewStore("/nonexistent/path/kanban.db")
	snapshot := store.Load()

	if snapshot.LoadError == nil {
		t.Fatal("expected error for non-existent DB, got nil")
	}

	// Should still have placeholder columns.
	if len(snapshot.Columns) == 0 {
		t.Fatal("expected fallback columns, got none")
	}
}

// --- Timestamp handling tests ---

func TestLoadTaskTimestamps(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	// Insert with specific timestamps.
	if store.db == nil {
		t.Fatal("no db")
	}
	created := int64(1000000)
	started := int64(2000000)
	completed := int64(3000000)
	heartbeat := int64(4000000)
	_, err := store.db.Exec(
		"INSERT INTO tasks (id, title, assignee, status, priority, created_at, started_at, completed_at, last_heartbeat_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
		"t1", "Timestamp Task", "coder", "done", 0, created, started, completed, heartbeat,
	)
	if err != nil {
		t.Fatalf("insert: %v", err)
	}

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	all := snapshot.AllTasks()
	if len(all) != 1 {
		t.Fatalf("AllTasks() = %d, want 1", len(all))
	}

	task := all[0]
	if task.CreatedAt.Unix() != created {
		t.Errorf("CreatedAt = %v, want unix %d", task.CreatedAt, created)
	}
	// UpdatedAt should be the latest timestamp (heartbeat).
	if task.UpdatedAt.Unix() != heartbeat {
		t.Errorf("UpdatedAt = %v, want unix %d (heartbeat)", task.UpdatedAt, heartbeat)
	}
}

func TestLoadTaskUpdatedAtFromCompleted(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	if store.db == nil {
		t.Fatal("no db")
	}
	created := int64(1000000)
	completed := int64(3000000)
	_, err := store.db.Exec(
		"INSERT INTO tasks (id, title, assignee, status, priority, created_at, started_at, completed_at, last_heartbeat_at) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL)",
		"t1", "No Start", "coder", "done", 0, created, completed,
	)
	if err != nil {
		t.Fatalf("insert: %v", err)
	}

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	all := snapshot.AllTasks()
	if len(all) != 1 {
		t.Fatalf("AllTasks() = %d, want 1", len(all))
	}

	// UpdatedAt should be completed_at since there's no started_at or heartbeat.
	if all[0].UpdatedAt.Unix() != completed {
		t.Errorf("UpdatedAt = %v, want unix %d (completed)", all[0].UpdatedAt, completed)
	}
}

func TestLoadTaskUpdatedAtFromCreatedOnly(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	if store.db == nil {
		t.Fatal("no db")
	}
	created := int64(1000000)
	_, err := store.db.Exec(
		"INSERT INTO tasks (id, title, assignee, status, priority, created_at, started_at, completed_at, last_heartbeat_at) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)",
		"t1", "Fresh", "coder", "todo", 0, created,
	)
	if err != nil {
		t.Fatalf("insert: %v", err)
	}

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	all := snapshot.AllTasks()
	if len(all) != 1 {
		t.Fatalf("AllTasks() = %d, want 1", len(all))
	}

	// UpdatedAt should fall back to created_at.
	if all[0].UpdatedAt.Unix() != created {
		t.Errorf("UpdatedAt = %v, want unix %d (created)", all[0].UpdatedAt, created)
	}
}

// --- Priority field test ---

func TestLoadTaskPriority(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	insertTask(t, store, "t1", "Low Priority", "coder", "todo", 0)
	insertTask(t, store, "t2", "High Priority", "coder", "todo", 5)

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	all := snapshot.AllTasks()
	if len(all) != 2 {
		t.Fatalf("AllTasks() = %d, want 2", len(all))
	}

	for _, task := range all {
		if task.Title == "High Priority" && task.Priority != 5 {
			t.Errorf("High Priority task Priority = %d, want 5", task.Priority)
		}
		if task.Title == "Low Priority" && task.Priority != 0 {
			t.Errorf("Low Priority task Priority = %d, want 0", task.Priority)
		}
	}
}

// --- SQL error handling ---

func TestLoadQueryError(t *testing.T) {
	store := newInMemoryStore(t)
	defer store.Close()
	// Don't create the tasks table - query should fail.

	snapshot := store.Load()
	if snapshot.LoadError == nil {
		t.Fatal("expected error when tasks table missing, got nil")
	}

	// Should still have fallback columns.
	if len(snapshot.Columns) == 0 {
		t.Fatal("expected fallback columns on query error, got none")
	}
}

// --- Store.Close idempotency ---

func TestStoreCloseIdempotent(t *testing.T) {
	store := newInMemoryStore(t)

	err1 := store.Close()
	if err1 != nil {
		t.Errorf("first Close() = %v, want nil", err1)
	}

	// Second close on already-closed db may or may not error;
	// we just verify it doesn't panic.
	err2 := store.Close()
	_ = err2 // Accept either nil or "already closed" - both are fine.
}

// --- ColumnsFromTasks with unknown status ---

func TestColumnsFromTasksUnknownStatus(t *testing.T) {
	// Tasks with an unknown status should still be grouped (in a map bucket)
	// but won't appear in any canonical column.
	tasks := []Task{
		{ID: "t1", Title: "Known", Status: StatusTodo},
		{ID: "t2", Title: "Unknown", Status: TaskStatus("weird_status")},
	}

	columns := columnsFromTasks(tasks)

	// Canonical columns should all be present.
	if len(columns) != len(ValidStatuses()) {
		t.Errorf("columns = %d, want %d", len(columns), len(ValidStatuses()))
	}

	// Only the "todo" column should have a task.
	found := 0
	for _, col := range columns {
		found += len(col.Tasks)
	}
	// The unknown status task won't appear in any canonical column.
	if found != 1 {
		t.Errorf("tasks in canonical columns = %d, want 1 (unknown status excluded)", found)
	}
}

// --- Verify all status constants appear in SQL CASE ---

func TestStatusConstantsMatchSQL(t *testing.T) {
	// This test verifies that the status values used in ValidStatuses()
	// match the CASE WHEN values in the SQL query in loadTasks().
	// If someone adds a new status constant but forgets the SQL, this catches it.
	// We verify by loading tasks with every status and confirming they all land in columns.

	store := newInMemoryStore(t)
	defer store.Close()
	setupTestSchema(t, store)

	for _, status := range ValidStatuses() {
		insertTask(t, store, fmt.Sprintf("t_%s", status), string(status)+" task", "coder", string(status), 0)
	}

	snapshot := store.Load()
	if snapshot.LoadError != nil {
		t.Fatalf("Load() error: %v", snapshot.LoadError)
	}

	// Every column should have exactly one task.
	for _, col := range snapshot.Columns {
		if len(col.Tasks) != 1 {
			t.Errorf("column %q has %d tasks, want 1", col.Status, len(col.Tasks))
		}
	}
}
