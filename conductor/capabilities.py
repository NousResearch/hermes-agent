"""Static capability boundaries for governed conductor roles."""


def conductor_capabilities() -> frozenset[str]:
    return frozenset({"dispatch", "observe", "verify_receipt"})


def reviewer_capabilities() -> frozenset[str]:
    return frozenset({"read", "review", "write_receipt"})
