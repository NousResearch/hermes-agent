package kanbanui

import (
	"database/sql"
	"fmt"
	"sort"
	"time"

	_ "modernc.org/sqlite"
)

// Store manages a SQLite connection to the Hermes kanban database.
// It returns domain models (Task, BoardSnapshot, etc.) rather than raw SQL rows.
type Store struct {
	dbPath string
	db     *sql.DB
}

// NewStore opens a SQLite database at the given path and returns a Store handle.
// Returns an error if the database cannot be opened.
func NewStore(dbPath string) Store {
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		return Store{dbPath: dbPath}
	}
	// Verify the connection is alive.
	if err := db.Ping(); err != nil {
		return Store{dbPath: dbPath}
	}
	return Store{dbPath: dbPath, db: db}
}

// Close releases the database connection.
func (s Store) Close() error {
	if s.db != nil {
		return s.db.Close()
	}
	return nil
}

// Path returns the configured SQLite database path.
func (s Store) Path() string {
	return s.dbPath
}

// Load returns a BoardSnapshot with tasks grouped by status into columns.
// Within each column, tasks are sorted by assignee (ascending) then title (ascending).
// Returns a snapshot with LoadError set if the database is unavailable.
func (s Store) Load() BoardSnapshot {
	snapshot := BoardSnapshot{
		LoadedAt: time.Now(),
		Source:   s.dbPath,
	}

	if s.db == nil {
		snapshot.LoadError = fmt.Errorf("store: database not initialized (path: %s)", s.dbPath)
		snapshot.Columns = columnsFromTasks(PlaceholderTasks())
		return snapshot
	}

	tasks, err := s.loadTasks()
	if err != nil {
		snapshot.LoadError = fmt.Errorf("store: failed to load tasks: %w", err)
		snapshot.Columns = columnsFromTasks(PlaceholderTasks())
		return snapshot
	}

	snapshot.Columns = columnsFromTasks(tasks)
	return snapshot
}

// loadTasks queries all tasks from the database and returns them as domain models.
func (s Store) loadTasks() ([]Task, error) {
	rows, err := s.db.Query(`
		SELECT id, title, assignee, status, priority, created_at, started_at, completed_at, last_heartbeat_at
		FROM tasks
		ORDER BY
			CASE status
				WHEN 'triage' THEN 0
				WHEN 'todo' THEN 1
				WHEN 'ready' THEN 2
				WHEN 'running' THEN 3
				WHEN 'blocked' THEN 4
				WHEN 'done' THEN 5
				ELSE 6
			END,
			COALESCE(assignee, '') ASC,
			title ASC
	`,
	)
	if err != nil {
		return nil, fmt.Errorf("query tasks: %w", err)
	}
	defer rows.Close()

	var tasks []Task
	for rows.Next() {
		var t Task
		var assignee sql.NullString
		var createdAt, startedAt, completedAt, heartbeatAt sql.NullInt64

		err := rows.Scan(
			&t.ID, &t.Title, &assignee, &t.Status, &t.Priority,
			&createdAt, &startedAt, &completedAt, &heartbeatAt,
		)
		if err != nil {
			return nil, fmt.Errorf("scan task row: %w", err)
		}

		if assignee.Valid {
			t.Assignee = assignee.String
		}
		if createdAt.Valid {
			t.CreatedAt = time.Unix(createdAt.Int64, 0)
			t.UpdatedAt = time.Unix(createdAt.Int64, 0)
		}
		// Derive UpdatedAt from the most recent activity timestamp.
		if startedAt.Valid {
			t.UpdatedAt = time.Unix(startedAt.Int64, 0)
		}
		if completedAt.Valid {
			t.UpdatedAt = time.Unix(completedAt.Int64, 0)
		}
		if heartbeatAt.Valid {
			ts := time.Unix(heartbeatAt.Int64, 0)
			if ts.After(t.UpdatedAt) {
				t.UpdatedAt = ts
			}
		}

		tasks = append(tasks, t)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate task rows: %w", err)
	}

	return tasks, nil
}

// UpdateTaskStatus changes the status of a task by ID.
// Returns an error if the task is not found or the status is invalid.
func (s Store) UpdateTaskStatus(taskID string, newStatus TaskStatus) error {
	if s.db == nil {
		return fmt.Errorf("store: database not initialized")
	}

	// Validate status.
	var valid bool
	for _, vs := range ValidStatuses() {
		if newStatus == vs {
			valid = true
			break
		}
	}
	if !valid {
		return fmt.Errorf("store: invalid status %q", newStatus)
	}

	result, err := s.db.Exec(
		"UPDATE tasks SET status = ? WHERE id = ?",
		newStatus, taskID,
	)
	if err != nil {
		return fmt.Errorf("update task status: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("check update result: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("store: task %q not found", taskID)
	}

	return nil
}

// UpdateTaskAssignee changes the assignee of a task by ID.
// Returns an error if the task is not found.
func (s Store) UpdateTaskAssignee(taskID string, assignee string) error {
	if s.db == nil {
		return fmt.Errorf("store: database not initialized")
	}

	result, err := s.db.Exec(
		"UPDATE tasks SET assignee = ? WHERE id = ?",
		assignee, taskID,
	)
	if err != nil {
		return fmt.Errorf("update task assignee: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("check update result: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("store: task %q not found", taskID)
	}

	return nil
}

// GetTaskDetail returns the full detail for a single task including comments and parent/child links.
func (s Store) GetTaskDetail(taskID string) (TaskDetail, error) {
	var detail TaskDetail

	if s.db == nil {
		return detail, fmt.Errorf("store: database not initialized")
	}

	// Load the task row.
	var body, createdBy sql.NullString
	var startedAt, completedAt sql.NullInt64
	var createdAt int64

	err := s.db.QueryRow(`
		SELECT id, title, body, assignee, status, priority, created_by, created_at, started_at, completed_at
		FROM tasks WHERE id = ?
	`, taskID).Scan(
		&detail.ID, &detail.Title, &body, &detail.Assignee,
		&detail.Status, &detail.Priority, &createdBy,
		&createdAt, &startedAt, &completedAt,
	)
	if err == sql.ErrNoRows {
		return detail, fmt.Errorf("store: task %q not found", taskID)
	}
	if err != nil {
		return detail, fmt.Errorf("query task detail: %w", err)
	}

	detail.CreatedAt = time.Unix(createdAt, 0)
	detail.UpdatedAt = detail.CreatedAt
	if body.Valid {
		detail.Body = body.String
	}
	if createdBy.Valid {
		detail.CreatedBy = createdBy.String
	}
	if startedAt.Valid {
		detail.UpdatedAt = time.Unix(startedAt.Int64, 0)
	}
	if completedAt.Valid {
		detail.UpdatedAt = time.Unix(completedAt.Int64, 0)
	}

	// Load comments.
	comments, err := s.loadComments(taskID)
	if err != nil {
		return detail, fmt.Errorf("load comments: %w", err)
	}
	detail.Comments = comments

	// Load parent links.
	parents, err := s.loadParentIDs(taskID)
	if err != nil {
		return detail, fmt.Errorf("load parents: %w", err)
	}
	detail.ParentIDs = parents

	// Load child links.
	children, err := s.loadChildIDs(taskID)
	if err != nil {
		return detail, fmt.Errorf("load children: %w", err)
	}
	detail.ChildIDs = children

	return detail, nil
}

// loadComments fetches all comments for a given task.
func (s Store) loadComments(taskID string) ([]Comment, error) {
	rows, err := s.db.Query(`
		SELECT id, task_id, author, body, created_at
		FROM task_comments WHERE task_id = ?
		ORDER BY created_at ASC
	`, taskID)
	if err != nil {
		return nil, fmt.Errorf("query comments: %w", err)
	}
	defer rows.Close()

	var comments []Comment
	for rows.Next() {
		var c Comment
		var ts int64
		err := rows.Scan(&c.ID, &c.TaskID, &c.Author, &c.Body, &ts)
		if err != nil {
			return nil, fmt.Errorf("scan comment: %w", err)
		}
		c.CreatedAt = time.Unix(ts, 0)
		comments = append(comments, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate comments: %w", err)
	}
	return comments, nil
}

// loadParentIDs returns the IDs of all parent tasks for a given task.
func (s Store) loadParentIDs(taskID string) ([]string, error) {
	rows, err := s.db.Query(`
		SELECT parent_id FROM task_links WHERE child_id = ?
	`, taskID)
	if err != nil {
		return nil, fmt.Errorf("query parent links: %w", err)
	}
	defer rows.Close()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("scan parent link: %w", err)
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// loadChildIDs returns the IDs of all child tasks for a given task.
func (s Store) loadChildIDs(taskID string) ([]string, error) {
	rows, err := s.db.Query(`
		SELECT child_id FROM task_links WHERE parent_id = ?
	`, taskID)
	if err != nil {
		return nil, fmt.Errorf("query child links: %w", err)
	}
	defer rows.Close()

	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("scan child link: %w", err)
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// columnsFromTasks groups a flat task slice into Columns by status,
// preserving the canonical column order defined by ValidStatuses().
// Tasks within each column are already sorted by the SQL query.
func columnsFromTasks(tasks []Task) []Column {
	// Build a map from status to tasks.
	byStatus := make(map[TaskStatus][]Task)
	for _, t := range tasks {
		byStatus[t.Status] = append(byStatus[t.Status], t)
	}

	// Fill in canonical order. Include empty columns so the board always shows all statuses.
	var columns []Column
	for _, status := range ValidStatuses() {
		col := Column{
			Status: status,
			Tasks:  byStatus[status],
		}
		columns = append(columns, col)
	}

	return columns
}

// sortTasksByAssigneeTitle sorts tasks in-place by assignee then title.
// Used when tasks come from sources that don't pre-sort.
func sortTasksByAssigneeTitle(tasks []Task) {
	sort.Slice(tasks, func(i, j int) bool {
		if tasks[i].Assignee != tasks[j].Assignee {
			return tasks[i].Assignee < tasks[j].Assignee
		}
		return tasks[i].Title < tasks[j].Title
	})
}

// LoadTaskEvents returns all lifecycle events for a task after the given since time.
// If since is the zero time, returns all events.
// Returns a slice sorted by created_at ascending.
func (s Store) LoadTaskEvents(taskID string, since time.Time) []TaskEvent {
	if s.db == nil {
		return nil
	}

	var query string
	var args []interface{}

	if since.IsZero() {
		query = `
			SELECT task_id, kind, payload, created_at, run_id
			FROM task_events WHERE task_id = ?
			ORDER BY created_at ASC
		`
		args = []interface{}{taskID}
	} else {
		query = `
			SELECT task_id, kind, payload, created_at, run_id
			FROM task_events WHERE task_id = ? AND created_at > ?
			ORDER BY created_at ASC
		`
		args = []interface{}{taskID, since.Unix()}
	}

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil
	}
	defer rows.Close()

	var events []TaskEvent
	for rows.Next() {
		var ev TaskEvent
		var payload sql.NullString
		var ts int64
		var runID sql.NullInt64

		err := rows.Scan(&ev.TaskID, &ev.Kind, &payload, &ts, &runID)
		if err != nil {
			continue
		}
		ev.CreatedAt = time.Unix(ts, 0)
		if payload.Valid {
			ev.Payload = payload.String
		}
		if runID.Valid {
			ev.RunID = int(runID.Int64)
		}
		events = append(events, ev)
	}

	return events
}
