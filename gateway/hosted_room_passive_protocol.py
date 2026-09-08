"""Passive evidence formats are response siblings, not RoomLink catalog fields."""


def passive_capabilities():
    # Work-record v2 is owned by #104759, not available on this lower baseline.
    return {"history_versions": [1, 2], "retirement_versions": [1, 2], "work_record_versions": []}


def supports_lineage(response):
    """Unknown/malformed capability versions are not permission to downgrade."""
    value = response.get("passive_replication") if isinstance(response, dict) else None
    if not isinstance(value, dict) or set(value) != set(passive_capabilities()):
        return False
    for versions in value.values():
        if (not isinstance(versions, list) or len(versions) > 2
                or any(type(v) is not int or v not in {1, 2} for v in versions)
                or len(set(versions)) != len(versions)):
            return False
    return 2 in value["history_versions"] and 2 in value["retirement_versions"]
