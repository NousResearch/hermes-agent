"""Old module name -> current module name.

A key is added when a module moves and is never removed; when the target moves again, only its
value changes. Plain dicts and no imports, because a process from an older release reads this file.
"""

MOVES: dict[str, str] = {
    "batch_runner": "hermes_agent.batch_runner",
    "toolset_distributions": "hermes_agent.toolset_distributions",
    "trajectory_compressor": "hermes_agent.trajectory_compressor",
}
