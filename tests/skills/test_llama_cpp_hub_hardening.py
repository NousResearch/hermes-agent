from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / "skills" / "mlops" / "inference" / "llama-cpp"


def _read(relative_path: str) -> str:
    return (SKILL_DIR / relative_path).read_text(encoding="utf-8")


def test_hub_discovery_requires_validation_and_shell_escaping() -> None:
    """HF repo metadata is attacker controlled and must not be spliced into shell commands."""
    hub_discovery = _read("references/hub-discovery.md")

    assert "Treat fetched page snippets, repo IDs, quant labels, and tree API `path` values as untrusted" in hub_discovery
    assert "Reject values containing shell metacharacters, quotes, whitespace, or control characters" in hub_discovery
    assert "Shell-escape each accepted repo ID, quant label, and GGUF path with `shlex.quote`" in hub_discovery
    assert "llama-server --hf-repo <repo> --hf-file <filename.gguf>" not in hub_discovery


def test_llama_cpp_skill_does_not_render_raw_hf_file_command() -> None:
    skill = _read("SKILL.md")

    assert "path` and `size` as untrusted source data" in skill
    assert "shell-escape every repo ID, quant label, and file path with `shlex.quote`" in skill
    assert "llama-server --hf-repo <repo> --hf-file <filename.gguf>" not in skill
