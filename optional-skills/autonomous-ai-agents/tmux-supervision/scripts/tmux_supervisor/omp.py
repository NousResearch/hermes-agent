"""OMP-specific launch options and lifecycle extension adapter."""

from pathlib import Path
import os

from .supervision import _binding, _require, _read_bytes, _text, _executable
from .supervision import launch as launch_tmux

EXTENSION = Path(__file__).with_name("omp_extension.ts")
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh", "max", "auto")


def launch(
    run_dir,
    prompt_file,
    omp_executable="omp",
    tmux_executable="tmux",
    canary=False,
    *,
    model=None,
    thinking=None,
    append_system_prompt=None,
):
    run, binding = _binding(run_dir)
    _require(binding["adapter"] == "omp", "adapter_mismatch")
    prompt = Path(prompt_file).absolute()
    _require(
        _read_bytes(prompt, private=False, limit=1024 * 1024).strip(), "empty_prompt"
    )
    options = []
    if model is not None:
        _require(_text(model) and model.strip(), "invalid_model")
        options.append("--model=" + model)
    if thinking is not None:
        _require(
            isinstance(thinking, str) and thinking in THINKING_LEVELS,
            "invalid_thinking",
        )
        options.append("--thinking=" + thinking)
    if append_system_prompt is not None:
        _require(
            isinstance(append_system_prompt, (str, os.PathLike)),
            "invalid_system_prompt",
        )
        _require(_text(os.fspath(append_system_prompt)), "invalid_system_prompt")
        system_prompt = Path(append_system_prompt).absolute()
        _require(system_prompt == system_prompt.resolve(), "invalid_system_prompt")
        _require(
            _read_bytes(system_prompt, private=False, limit=1024 * 1024).strip(),
            "empty_system_prompt",
        )
        options += ["--append-system-prompt", str(system_prompt)]
    _require(
        _read_bytes(EXTENSION, private=False, limit=1024 * 1024).strip(),
        "missing_extension",
    )
    omp = _executable(omp_executable)
    argv = [
        "env",
        "OMP_HERMES_BINDING_FILE=" + str(run / "binding.json"),
        omp,
        *options,
        "-e",
        str(EXTENSION),
    ]
    if canary:
        argv += [
            "--no-tools",
            "--no-skills",
            "--no-rules",
            "--no-extensions",
            "--no-title",
            "--no-session",
            "--no-lsp",
        ]
    argv.append("@" + str(prompt))
    return launch_tmux(run, argv, tmux_executable)
