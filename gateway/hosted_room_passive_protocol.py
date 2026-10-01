"""Passive copy formats a participant accepts; response siblings, not RoomLink catalog fields."""


def passive_capabilities():
    return {"history_versions": [1], "retirement_versions": [1], "work_record_versions": [1]}
