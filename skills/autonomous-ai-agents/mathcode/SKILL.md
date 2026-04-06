---
name: mathcode
description: Formalize natural language math problems into Lean 4 theorems and prove them using the MathCode CLI agent. Use when users ask to mathematically prove a concept, formalize a lemma, or interact with Lean 4.
version: 1.0.0
---

# MathCode Formalization Skill

MathCode is a terminal-based AI coding assistant equipped with a built-in math formalization engine. It bridges the gap between natural language math problems and formal verification in Lean 4.

## Prerequisites
The user must have MathCode installed. If missing, clone and run setup:

    git clone https://github.com/math-ai-org/mathcode.git ~/mathcode
    cd ~/mathcode && bash setup.sh

*Note: Requires an Anthropic API key in `~/mathcode/.env` if Codex is not used.*

## Usage
To formalize and prove a theorem, use the `-p` flag and pipe the natural language problem:

    cd ~/mathcode
    echo "Prove that the product of two reflections is block-diagonalizable..." | ./run -p

The output Lean 4 files will be generated in `~/mathcode/LeanFormalizations/`.

