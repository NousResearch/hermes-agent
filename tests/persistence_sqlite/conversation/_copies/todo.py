from . import Copy

COPY = Copy(
    legacy='hermes_state_todo',
    doc='The latest todo tool result of a session.',
    names=(
        '_TODO_TOOLS',
    ),
    legacy_class='SessionTodoResultMixin',
    members=(
        'get_latest_todo_result',
        '_todo_copies',
        '_removed_todo_copies',
        '_todo_copy_removed',
    ),
)
