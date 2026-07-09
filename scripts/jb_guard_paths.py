#!/usr/bin/env python3
"""Garde-fou de confinement du fork Jean-Billie (lane F1).

Vérifie qu'une liste de fichiers modifiés (arguments, ou stdin à défaut)
reste dans le périmètre autorisé décrit par ``.github/jb-allowed-paths.txt``.
Sort avec le code 1 en nommant chaque fichier hors périmètre.

Auto-protection : les fichiers du garde-fou lui-même (``SELF_PROTECTED``)
sont refusés en dur, même si un glob de l'allowlist les couvrait — leur
modification exige le label « jb-core-approved ». En CI (``GITHUB_ACTIONS``
posé), une liste stdin VIDE est une anomalie (une PR a toujours au moins un
fichier) : exit 2 plutôt que fail-open.

Utilisation :
    python3 scripts/jb_guard_paths.py fichier1 fichier2 ...
    git diff --no-renames --name-only origin/main...HEAD \
        | python3 scripts/jb_guard_paths.py

Stdlib uniquement (argparse, os, pathlib, re, sys) — aucune dépendance.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

DEFAULT_ALLOWLIST = (
    Path(__file__).resolve().parent.parent / ".github" / "jb-allowed-paths.txt"
)

# Fichiers du garde-fou : toujours refusés, quel que soit le contenu de
# l'allowlist (anti-neutralisation — revue B1). Les modifier = label
# « jb-core-approved » sur la PR.
SELF_PROTECTED = frozenset(
    {
        ".github/jb-allowed-paths.txt",
        ".github/workflows/jb-guard.yml",
        "scripts/jb_guard_paths.py",
    }
)


def load_allowlist(path: Path | str) -> list[str]:
    """Lit l'allowlist : un glob par ligne, ignore vides et commentaires ``#``."""
    patterns: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Traduit un glob en regex ancrée.

    Sémantique gitignore-like : ``**`` traverse les séparateurs ``/``
    (``a/**/b`` matche ``a/b``, ``a/x/b``, ``a/x/y/b``) ; ``*`` et ``?``
    restent dans un segment (ne matchent pas ``/``). Plus strict que
    ``fnmatch`` (où ``*`` traverse les ``/``).
    """
    out: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        if pattern.startswith("**", i):
            at_segment_start = i == 0 or pattern[i - 1] == "/"
            followed_by_slash = pattern[i + 2 : i + 3] == "/"
            if at_segment_start and followed_by_slash:
                # « **/ » en tête ou « /**/ » : zéro segment ou plus,
                # slash compris — « a/**/b » matche donc aussi « a/b ».
                out.append("(?:[^/]+/)*")
                i += 3
            else:
                # « ** » libre ou final (« dir/** ») : traverse tout.
                out.append(".*")
                i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def normalize(path: str) -> str:
    """Normalise un chemin git : séparateurs ``/``, sans ``./`` de tête."""
    p = path.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def find_offenders(files: list[str], patterns: list[str]) -> list[str]:
    """Renvoie les fichiers qui ne matchent AUCUN glob de l'allowlist."""
    regexes = [glob_to_regex(p) for p in patterns]
    return [f for f in files if not any(r.match(f) for r in regexes)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Échoue (code 1) si un fichier modifié sort du périmètre autorisé "
            "du fork Jean-Billie."
        )
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Fichiers modifiés à vérifier (à défaut : lus sur stdin, un par ligne).",
    )
    parser.add_argument(
        "--allowlist",
        default=str(DEFAULT_ALLOWLIST),
        help="Chemin de l'allowlist de globs (défaut : .github/jb-allowed-paths.txt).",
    )
    args = parser.parse_args(argv)

    stdin_mode = not args.files
    raw_files = sys.stdin.read().splitlines() if stdin_mode else args.files
    files = [normalize(f) for f in raw_files if f.strip()]

    allowlist_path = Path(args.allowlist)
    if not allowlist_path.is_file():
        print(f"jb-guard: allowlist introuvable : {allowlist_path}", file=sys.stderr)
        return 2

    patterns = load_allowlist(allowlist_path)
    if not files:
        if stdin_mode and os.environ.get("GITHUB_ACTIONS"):
            # Anti fail-open (revue I3) : en CI, une PR a toujours au moins
            # un fichier modifié. Liste vide = le diff amont a échoué.
            print(
                "jb-guard: ÉCHEC — liste de fichiers VIDE sur stdin en CI : "
                "le `git diff` amont a probablement échoué (base introuvable, "
                "fetch incomplet…). Refus de conclure « rien à vérifier ».",
                file=sys.stderr,
            )
            return 2
        print("jb-guard: aucun fichier modifié — rien à vérifier.")
        return 0

    protected = [f for f in files if f in SELF_PROTECTED]
    offenders = find_offenders(
        [f for f in files if f not in SELF_PROTECTED], patterns
    )
    if protected or offenders:
        print("jb-guard: ÉCHEC —", file=sys.stderr)
        if protected:
            print(
                "  fichiers du garde-fou (auto-protégés, jamais "
                "allowlistables) :",
                file=sys.stderr,
            )
            for f in protected:
                print(f"    - {f}", file=sys.stderr)
        if offenders:
            print(
                f"  fichiers hors du périmètre autorisé ({allowlist_path.name}) :",
                file=sys.stderr,
            )
            for f in offenders:
                print(f"    - {f}", file=sys.stderr)
        print(
            "\nRègle : la divergence vit dans plugins/jb_outbound/ + les fichiers "
            "additifs déclarés. Zéro nouveau patch du cœur Hermes.\n"
            "Si cette modification est volontaire (résorption F2/F3, évolution "
            "du garde-fou), posez le label « jb-core-approved » sur la PR.",
            file=sys.stderr,
        )
        return 1

    print(f"jb-guard: OK — {len(files)} fichier(s) dans le périmètre autorisé.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
