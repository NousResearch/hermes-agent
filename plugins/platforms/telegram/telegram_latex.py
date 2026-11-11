"""LaTeX macros in prose become the Unicode symbols they name.

Models such as gemma-4 write ``A \\rightarrow B`` or ``$x \\leq y$`` in ordinary chat replies.
MarkdownV2 has no math, so Telegram showed the raw backslash macro. Macros outside code are mapped
to their symbol, and an inline ``$...$`` span that held one loses its now-pointless dollar signs
once no macro is left in it. Rich messages typeset math themselves, so there only macros outside
every math span are mapped. Unknown macros, code, and paths (``C:\\in``, ``a/\\to``) stay as written.
"""

import re

_MACROS = {
    # arrows
    "to": "→", "rightarrow": "→", "leftarrow": "←", "gets": "←", "leftrightarrow": "↔",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "Leftrightarrow": "⇔", "longrightarrow": "⟶",
    "longleftarrow": "⟵", "Longrightarrow": "⟹", "Longleftarrow": "⟸", "implies": "⟹",
    "iff": "⟺", "mapsto": "↦", "uparrow": "↑", "downarrow": "↓", "nearrow": "↗", "searrow": "↘",
    # relations
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠", "approx": "≈",
    "equiv": "≡", "sim": "∼", "simeq": "≃", "cong": "≅", "propto": "∝", "ll": "≪", "gg": "≫",
    "perp": "⊥", "parallel": "∥",
    # operators
    "times": "×", "div": "÷", "pm": "±", "mp": "∓", "cdot": "·", "circ": "∘", "bullet": "•",
    "star": "⋆", "oplus": "⊕", "otimes": "⊗",
    # sets and logic
    "in": "∈", "notin": "∉", "ni": "∋", "subset": "⊂", "supset": "⊃", "subseteq": "⊆",
    "supseteq": "⊇", "cup": "∪", "cap": "∩", "forall": "∀", "exists": "∃", "nexists": "∄",
    "emptyset": "∅", "varnothing": "∅", "setminus": "∖", "land": "∧", "wedge": "∧", "lor": "∨",
    "vee": "∨", "neg": "¬", "lnot": "¬", "top": "⊤", "bot": "⊥",
    # calculus
    "infty": "∞", "partial": "∂", "nabla": "∇", "sum": "∑", "prod": "∏", "int": "∫",
    "oint": "∮", "sqrt": "√",
    # greek
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "varepsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ", "sigma": "σ",
    "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    # misc
    "ldots": "…", "cdots": "⋯", "dots": "…", "langle": "⟨", "rangle": "⟩", "lceil": "⌈",
    "rceil": "⌉", "lfloor": "⌊", "rfloor": "⌋", "angle": "∠", "hbar": "ℏ", "ell": "ℓ",
    "aleph": "ℵ", "dagger": "†", "checkmark": "✓", "degree": "°",
}
# A macro is a backslash word that no letter extends, never part of a path or an escaped backslash.
_MACRO_NAMES = "|".join(sorted(_MACROS, key=len, reverse=True))
_MACRO_RE = re.compile(r"(?<![\w:/\\])\\(" + _MACRO_NAMES + r")(?![A-Za-z])")
# Inside a confirmed ``$...$`` span a macro may follow a letter (``$x\leq y$``): no path guard there.
_MATH_MACRO_RE = re.compile(r"(?<!\\)\\(" + _MACRO_NAMES + r")(?![A-Za-z])")
_INLINE_MATH_RE = re.compile(r"(?<![\\$\w])\$([^$\n]{1,120}?)\$(?![\w$])")


def _replace_macros(text: str) -> str:
    return _MACRO_RE.sub(lambda m: _MACROS[m.group(1)], text)


def _inline_math(m: re.Match) -> str:
    body = m.group(1)
    if not _MATH_MACRO_RE.search(body):
        return m.group(0)  # no known macro: an amount pair or real math we can't render; leave it
    converted = _MATH_MACRO_RE.sub(lambda mm: _MACROS[mm.group(1)], body)
    return converted.strip() if "\\" not in converted and "{" not in converted else f"${converted}$"


def normalize_latex_macros(prose: str) -> str:
    """Map known LaTeX macros in *prose* (code already excluded by the caller) to Unicode."""
    if "\\" not in prose:
        return prose
    return _replace_macros(_INLINE_MATH_RE.sub(_inline_math, prose))


def replace_macros_outside(prose: str, kept_spans) -> str:
    """Rich-message variant: map known macros in *prose* outside *kept_spans* (the math spans the
    rich client typesets itself, sorted ``(start, end)``; code is already excluded by the caller)."""
    out, pos = [], 0
    for start, end in kept_spans:
        if start < pos:
            continue
        out.append(_replace_macros(prose[pos:start]))
        out.append(prose[start:end])
        pos = end
    out.append(_replace_macros(prose[pos:]))
    return "".join(out)
