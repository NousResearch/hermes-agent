"""Kynver AgentOS tool schemas exposed through the memory provider."""

MEMORY_SEARCH_SCHEMA = {
    "name": "kynver_memory_search",
    "description": "Search authoritative Kynver AgentOS memory/context.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language memory query."},
            "k": {"type": "integer", "description": "Maximum results, default 5, max 20."},
        },
        "required": ["query"],
    },
}

MEMORY_WRITE_SCHEMA = {
    "name": "kynver_memory_write",
    "description": "Write durable memory to Kynver AgentOS with Hermes Forge provenance.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Memory content to store."},
            "key": {"type": "string", "description": "Optional stable memory key."},
            "memoryType": {"type": "string", "description": "fact, decision, preference, lesson, or runbook."},
        },
        "required": ["content"],
    },
}

TASK_CREATE_SCHEMA = {
    "name": "kynver_task_create",
    "description": "Create a Kynver AgentOS task/control-plane record.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short task title."},
            "description": {"type": "string", "description": "Task details."},
            "priority": {"type": "string", "description": "low, normal, high, or critical."},
            "executor": {"type": "string", "description": "inline, harness, acp, or manual."},
            "executorRef": {"type": "string"},
            "parentTaskId": {"type": "string"},
            "goalId": {"type": "string"},
            "projectId": {"type": "string"},
            "personaSlug": {"type": "string"},
            "scheduledFor": {"type": "string", "description": "ISO-8601 scheduled start time."},
            "dependsOnTaskIds": {"type": "array", "items": {"type": "string"}},
            "idempotencyKey": {"type": "string", "description": "Optional stable idempotency key."},
            "requestId": {"type": "string", "description": "Raw request id used as a dedupe key by AgentOS."},
        },
        "required": ["title"],
    },
}

TASK_UPDATE_SCHEMA = {
    "name": "kynver_task_update",
    "description": "Patch status, priority, links, progress, or artifact refs on a Kynver task.",
    "parameters": {
        "type": "object",
        "properties": {
            "taskId": {"type": "string", "description": "Kynver task id."},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "status": {"type": "string", "description": "ready, running, waiting, scheduled, blocked, needs_input, awaiting_review, done, failed, or cancelled."},
            "priority": {"type": "string"},
            "executor": {"type": "string"},
            "executorRef": {"type": "string"},
            "parentTaskId": {"type": "string"},
            "goalId": {"type": "string"},
            "projectId": {"type": "string"},
            "personaSlug": {"type": "string"},
            "scheduledFor": {"type": "string"},
            "dependsOnTaskIds": {"type": "array", "items": {"type": "string"}},
            "lastSummary": {"type": "string"},
            "blocker": {"type": "string"},
            "branch": {"type": "string"},
            "worktreePath": {"type": "string"},
            "prUrl": {"type": "string"},
            "headCommit": {"type": "string"},
        },
        "required": ["taskId"],
    },
}

TASK_LIST_SCHEMA = {
    "name": "kynver_task_list",
    "description": "List Kynver AgentOS tasks.",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "description": "Optional status filter."},
            "limit": {"type": "integer", "description": "Maximum tasks, default 20."},
        },
    },
}

TASK_CLOSE_SCHEMA = {
    "name": "kynver_task_close",
    "description": "Close a Kynver AgentOS task with a terminal status.",
    "parameters": {
        "type": "object",
        "properties": {
            "taskId": {"type": "string"},
            "status": {"type": "string", "description": "done, failed, or cancelled. Defaults to done."},
            "summary": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["taskId"],
    },
}

TASK_LOG_EVENT_SCHEMA = {
    "name": "kynver_task_log_event",
    "description": "Append an audit/progress event to a Kynver AgentOS task.",
    "parameters": {
        "type": "object",
        "properties": {
            "taskId": {"type": "string"},
            "eventType": {"type": "string", "description": "created, started, worker_update, blocked, steer, artifact, review, done, or failed."},
            "message": {"type": "string"},
            "payload": {"type": "object"},
            "artifactVisibility": {"type": "string"},
            "eventKey": {"type": "string"},
            "metadata": {"type": "object"},
        },
        "required": ["taskId", "eventType"],
    },
}

TASK_STEER_SCHEMA = {
    "name": "kynver_task_steer",
    "description": "Send a steering artifact to a Kynver AgentOS task if supported.",
    "parameters": {
        "type": "object",
        "properties": {
            "taskId": {"type": "string"},
            "message": {"type": "string"},
            "detail": {"type": "object"},
            "eventKey": {"type": "string"},
            "metadata": {"type": "object"},
        },
        "required": ["taskId", "message"],
    },
}

SKILL_LIST_SCHEMA = {
    "name": "kynver_skill_list",
    "description": "List Kynver AgentOS skill manifests without fetching full bodies.",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Optional category filter."},
            "limit": {"type": "integer", "description": "Maximum manifests, default 50."},
        },
    },
}

SKILL_SEARCH_SCHEMA = {
    "name": "kynver_skill_search",
    "description": "Filter Kynver AgentOS skill manifests client-side.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "description": "Maximum manifests, default 20."},
        },
        "required": ["query"],
    },
}

SKILL_GET_SCHEMA = {
    "name": "kynver_skill_get",
    "description": "Fetch a full Kynver skill body on demand as external user-authored content.",
    "parameters": {
        "type": "object",
        "properties": {
            "skillId": {"type": "string", "description": "Skill id, slug, or name from Kynver manifests."},
            "source": {"type": "string", "description": "Optional builtin or user source."},
        },
        "required": ["skillId"],
    },
}

ALL_TOOL_SCHEMAS = [
    MEMORY_SEARCH_SCHEMA,
    MEMORY_WRITE_SCHEMA,
    TASK_CREATE_SCHEMA,
    TASK_UPDATE_SCHEMA,
    TASK_LIST_SCHEMA,
    TASK_CLOSE_SCHEMA,
    TASK_LOG_EVENT_SCHEMA,
    TASK_STEER_SCHEMA,
    SKILL_LIST_SCHEMA,
    SKILL_SEARCH_SCHEMA,
    SKILL_GET_SCHEMA,
]
