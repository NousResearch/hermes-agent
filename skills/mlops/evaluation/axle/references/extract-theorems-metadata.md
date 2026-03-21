# Extract-Theorems Metadata Reference

Each theorem extracted by `extract_theorems` includes 20+ metadata fields in the `Document` object.

## Identity & positioning

| Field | Type | Description |
|---|---|---|
| `name` | str | Theorem name |
| `declaration` | str | Raw source code of the declaration |
| `signature` | str | Everything before proof body (e.g., "theorem foo : 1 = 1") |
| `type` | str | Pretty-printed type (e.g., "∀ (a b : ℕ), a + b = b + a") |
| `type_hash` | int | Canonical alpha-invariant type hash (for deduplication) |
| `index` | int | 0-based position in original file (non-contiguous for mutual defs) |
| `line_pos` | int | 1-based start line number |
| `end_line_pos` | int | 1-based end line number |

## Content

| Field | Type | Description |
|---|---|---|
| `content` | str | Self-contained Lean code (independently compilable with imports) |
| `tokens` | list[str] | Tokenized source (e.g., ["theorem", "foo", ":", ...]) |
| `is_sorry` | bool | Whether proof contains sorry |

## Proof complexity

| Field | Type | Description |
|---|---|---|
| `proof_length` | int | Approximate number of tactics |
| `tactic_counts` | dict[str, int] | Per-tactic breakdown (e.g., {"Lean.Parser.Tactic.omega": 1}) |

## Dependency analysis (6 lists)

| Field | Type | Description |
|---|---|---|
| `local_type_dependencies` | list[str] | Transitive local deps of the TYPE |
| `local_value_dependencies` | list[str] | Transitive local deps of the PROOF |
| `external_type_dependencies` | list[str] | Immediate external deps of the TYPE |
| `external_value_dependencies` | list[str] | Immediate external deps of the PROOF |
| `local_syntactic_dependencies` | list[str] | Local constants literally written in source |
| `external_syntactic_dependencies` | list[str] | External constants literally written in source |

"Syntactic" dependencies are what appears in source code before notation/macro expansion. "Type/value" dependencies include things that emerge from expansion.

## Compilation status

| Field | Type | Description |
|---|---|---|
| `document_messages` | Messages | {errors, warnings, infos} from standalone compilation |
| `theorem_messages` | Messages | {errors, warnings, infos} specific to this theorem |

## Example output

```json
{
  "add_comm_test": {
    "declaration": "theorem add_comm_test : ∀ (a b : Nat), a + b = b + a := by\n  intros a b\n  omega",
    "content": "import Mathlib\n\ntheorem add_comm_test : ∀ (a b : Nat), a + b = b + a := by\n  intros a b\n  omega",
    "signature": "theorem add_comm_test : ∀ (a b : Nat), a + b = b + a",
    "type": "∀ (a b : ℕ), a + b = b + a",
    "type_hash": 1387679959,
    "index": 0,
    "line_pos": 3,
    "end_line_pos": 5,
    "is_sorry": false,
    "proof_length": 2,
    "tactic_counts": {
      "Lean.Parser.Tactic.intros": 1,
      "Lean.Parser.Tactic.omega": 1
    },
    "tokens": ["theorem", "add_comm_test", ":", "∀", "(", "a", "b", ":", "Nat", ")", ",", "a", "+", "b", "=", "b", "+", "a", ":=", "by", "intros", "a", "b", "omega"],
    "external_syntactic_dependencies": ["Nat"],
    "external_type_dependencies": ["Nat", "Eq", "HAdd.hAdd", "instHAdd", "instAddNat"],
    "external_value_dependencies": ["Nat", "Decidable.byContradiction", "Eq", "HAdd.hAdd", "instHAdd", "instAddNat", "instDecidableEqNat", "Not"],
    "local_syntactic_dependencies": [],
    "local_type_dependencies": [],
    "local_value_dependencies": ["add_comm_test._proof_1"],
    "document_messages": {"errors": [], "infos": [], "warnings": []},
    "theorem_messages": {"errors": [], "infos": [], "warnings": []}
  }
}
```

## CLI usage

```bash
axle extract-theorems --environment lean-4.28.0 --ignore-imports file.lean -d output/ -f
```

Flags: `-d DIR` / `--output-dir DIR` (default: `extract_theorems/`), `-f` / `--force` (overwrite).
No `--names` or `--indices` filtering — always extracts all theorems.
