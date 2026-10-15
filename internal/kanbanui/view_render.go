package kanbanui

import (
	"fmt"
	"strings"
)

func (m *Model) render() string {
	var b strings.Builder

	b.WriteString(m.styles.Header.Render("Hermes Kanban TUI"))
	b.WriteString("\n")
	b.WriteString(m.styles.Muted.Render(fmt.Sprintf("db: %s  refresh: %s", m.config.DBPath, formatRefreshInterval(m.config.RefreshInterval))))
	b.WriteString("\n\n")
	b.WriteString(m.renderTaskList())
	b.WriteString("\n\n")
	b.WriteString(m.styles.Muted.Render("↑/↓ navigate · l logs · C-r refresh · q quit"))
	if m.status != "" {
		b.WriteString("\n")
		b.WriteString(m.styles.Muted.Render(m.status))
	}

	return b.String()
}

func (m *Model) renderTaskList() string {
	if len(m.tasks) == 0 {
		return m.styles.Muted.Render("No tasks loaded yet.")
	}

	width := m.width
	if width <= 0 {
		width = 100
	}

	var lines []string
	lines = append(lines, m.styles.Header.Render("Tasks"))
	for i, task := range m.tasks {
		status := statusStyle(m.styles, task.Status).Render(string(task.Status))
		line := fmt.Sprintf("%s  %-10s  %-8s  p%-2d  %s",
			selector(i == m.cursor), task.ID, status, task.Priority, task.Title)
		line = truncate(line, width)
		if i == m.cursor {
			line = m.styles.Selected.Render(line)
		}
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func selector(selected bool) string {
	if selected {
		return ">"
	}
	return " "
}
