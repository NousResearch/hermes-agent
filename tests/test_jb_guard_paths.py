"""Tests du garde-fou de confinement du fork (scripts/jb_guard_paths.py, lane F1)."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "jb_guard_paths.py"
REAL_ALLOWLIST = REPO_ROOT / ".github" / "jb-allowed-paths.txt"


def _load_module():
    spec = importlib.util.spec_from_file_location("jb_guard_paths", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_module()


@pytest.fixture
def allowlist(tmp_path: Path) -> Path:
    """Allowlist minimale de test, avec commentaires et lignes vides."""
    path = tmp_path / "allowed.txt"
    path.write_text(
        "# commentaire d'en-tête\n"
        "\n"
        "plugins/jb_outbound/**\n"
        "  # commentaire indenté\n"
        "Dockerfile\n"
        "tests/**\n"
        "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def hors_ci(monkeypatch) -> None:
    """Neutralise GITHUB_ACTIONS : ces tests valident le comportement LOCAL,
    ils doivent passer aussi quand la suite tourne… dans GitHub Actions."""
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)


# ---------------------------------------------------------------------------
# Parsing de l'allowlist
# ---------------------------------------------------------------------------


def test_load_allowlist_ignore_commentaires_et_lignes_vides(allowlist: Path):
    patterns = guard.load_allowlist(allowlist)
    assert patterns == ["plugins/jb_outbound/**", "Dockerfile", "tests/**"]


# ---------------------------------------------------------------------------
# Matching des globs
# ---------------------------------------------------------------------------


def test_glob_double_etoile_traverse_les_repertoires():
    patterns = ["plugins/jb_outbound/**"]
    assert guard.find_offenders(["plugins/jb_outbound/hooks.py"], patterns) == []
    assert (
        guard.find_offenders(["plugins/jb_outbound/sub/deep/mod.py"], patterns) == []
    )
    # Un préfixe voisin ne matche pas.
    assert guard.find_offenders(["plugins/other/mod.py"], patterns) == [
        "plugins/other/mod.py"
    ]


def test_glob_double_etoile_median_matche_zero_segment():
    """Revue M2 : gitignore-like => « a/**/b » matche AUSSI « a/b »."""
    patterns = ["a/**/b"]
    assert guard.find_offenders(["a/b"], patterns) == []
    assert guard.find_offenders(["a/x/b"], patterns) == []
    assert guard.find_offenders(["a/x/y/b"], patterns) == []
    # Pas de matching fantaisiste sans les slashs.
    assert guard.find_offenders(["ab"], patterns) == ["ab"]
    assert guard.find_offenders(["a/xb"], patterns) == ["a/xb"]


def test_glob_double_etoile_en_tete():
    patterns = ["**/conftest.py"]
    assert guard.find_offenders(["conftest.py"], patterns) == []
    assert guard.find_offenders(["tests/deep/conftest.py"], patterns) == []
    assert guard.find_offenders(["tests/conftest_extra.py"], patterns) == [
        "tests/conftest_extra.py"
    ]


def test_glob_simple_etoile_ne_traverse_pas_les_repertoires():
    patterns = ["plugins/*.py"]
    assert guard.find_offenders(["plugins/top.py"], patterns) == []
    assert guard.find_offenders(["plugins/sub/deep.py"], patterns) == [
        "plugins/sub/deep.py"
    ]


def test_fichier_exact_autorise():
    patterns = ["Dockerfile"]
    assert guard.find_offenders(["Dockerfile"], patterns) == []
    assert guard.find_offenders(["Dockerfile.dev"], patterns) == ["Dockerfile.dev"]


# ---------------------------------------------------------------------------
# CLI : codes de sortie et messages
# ---------------------------------------------------------------------------


def test_fichier_plugin_autorise_exit_0(allowlist: Path, capsys):
    code = guard.main(
        ["--allowlist", str(allowlist), "plugins/jb_outbound/hooks.py"]
    )
    assert code == 0
    assert "OK" in capsys.readouterr().out


def test_fichier_coeur_refuse_exit_1_et_nomme(allowlist: Path, capsys):
    code = guard.main(["--allowlist", str(allowlist), "gateway/run.py"])
    assert code == 1
    err = capsys.readouterr().err
    assert "gateway/run.py" in err
    assert "jb-core-approved" in err


def test_plusieurs_fichiers_dont_un_refuse(allowlist: Path, capsys):
    code = guard.main(
        [
            "--allowlist",
            str(allowlist),
            "plugins/jb_outbound/a.py",
            "tests/test_x.py",
            "cron/scheduler.py",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert "cron/scheduler.py" in err
    # Les fichiers autorisés ne sont PAS listés comme fautifs.
    assert "plugins/jb_outbound/a.py" not in err
    assert "tests/test_x.py" not in err


def test_stdin_et_normalisation_backslash(allowlist: Path, capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("plugins\\jb_outbound\\win.py\n\n./Dockerfile\n")
    )
    code = guard.main(["--allowlist", str(allowlist)])
    assert code == 0
    assert "2 fichier(s)" in capsys.readouterr().out


def test_aucun_fichier_exit_0_hors_ci(allowlist: Path, capsys, monkeypatch, hors_ci):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    code = guard.main(["--allowlist", str(allowlist)])
    assert code == 0
    assert "rien à vérifier" in capsys.readouterr().out


def test_stdin_vide_en_ci_exit_2(allowlist: Path, capsys, monkeypatch):
    """Revue I3 : en CI (GITHUB_ACTIONS), une liste vide = diff amont cassé
    => refus fail-closed, pas un « rien à vérifier » silencieux."""
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    code = guard.main(["--allowlist", str(allowlist)])
    assert code == 2
    err = capsys.readouterr().err
    assert "VIDE" in err
    assert "diff" in err


def test_allowlist_introuvable_exit_2(tmp_path: Path, capsys):
    code = guard.main(
        ["--allowlist", str(tmp_path / "absente.txt"), "plugins/jb_outbound/a.py"]
    )
    assert code == 2


# ---------------------------------------------------------------------------
# Auto-protection du garde-fou (revue B1)
# ---------------------------------------------------------------------------


def test_fichiers_du_garde_refuses_meme_si_allowlistes(tmp_path: Path, capsys):
    """La ceinture SELF_PROTECTED l'emporte : une PR qui ré-allowlisterait les
    fichiers du garde (.github/** par ex.) ne peut pas les libérer."""
    complice = tmp_path / "complice.txt"
    complice.write_text(".github/**\nscripts/**\n", encoding="utf-8")
    code = guard.main(
        [
            "--allowlist",
            str(complice),
            ".github/workflows/jb-guard.yml",
            ".github/jb-allowed-paths.txt",
            "scripts/jb_guard_paths.py",
        ]
    )
    assert code == 1
    err = capsys.readouterr().err
    assert ".github/workflows/jb-guard.yml" in err
    assert ".github/jb-allowed-paths.txt" in err
    assert "scripts/jb_guard_paths.py" in err
    assert "auto-protégés" in err
    assert "jb-core-approved" in err
    # Avec la même allowlist complice, un fichier .github NON garde passe :
    # c'est bien la ceinture qui refuse, pas l'allowlist.
    code = guard.main(["--allowlist", str(complice), ".github/dependabot.yml"])
    assert code == 0


# ---------------------------------------------------------------------------
# Allowlist réelle du repo (.github/jb-allowed-paths.txt)
# ---------------------------------------------------------------------------


def test_allowlist_reelle_perimetre(capsys):
    assert REAL_ALLOWLIST.is_file()
    patterns = guard.load_allowlist(REAL_ALLOWLIST)

    autorises = [
        "plugins/jb_outbound/deep/mod.py",
        "tests/test_jb_guard_paths.py",
        "gateway/platforms/api_server.py",
        "tools/request_tool_connection.py",
        "hermes_cli/mcp_config.py",
        "Dockerfile",
        "pyproject.toml",
        "tasks/notes.md",
    ]
    assert guard.find_offenders(autorises, patterns) == []

    # Revue I4 : plus de .github/** — les workflows UPSTREAM sont protégés.
    # Revue B1 : les fichiers du garde et scripts/ sont hors allowlist.
    refuses_perimetre = [
        ".github/workflows/lint.yml",
        ".github/workflows/tests.yml",
        ".github/workflows/jb-guard.yml",
        ".github/jb-allowed-paths.txt",
        "scripts/jb_guard_paths.py",
        "scripts/release.py",
    ]
    assert guard.find_offenders(refuses_perimetre, patterns) == refuses_perimetre

    # Les patchs cœur historiques restent REFUSÉS (dette F2/F3, échappatoire
    # = label jb-core-approved).
    refuses_coeur = [
        "gateway/run.py",
        "cron/scheduler.py",
        "tools/delegate_tool.py",
        "hermes_cli/config.py",
        "hermes_state.py",
        "hermes_cli/subcommands/mcp.py",
        "toolsets.py",
    ]
    assert guard.find_offenders(refuses_coeur, patterns) == refuses_coeur

    # Cohérence : SELF_PROTECTED ne doit matcher AUCUN glob de l'allowlist
    # réelle (la ceinture est en dur, l'allowlist ne les couvre pas non plus).
    assert guard.find_offenders(sorted(guard.SELF_PROTECTED), patterns) == sorted(
        guard.SELF_PROTECTED
    )
