---
name: mathcode
description: "Setup and usage of MathCode, the terminal-based mathematical coding agent and formalization engine for Lean 4. Covers Anthropic/Codex backend configuration, avoiding workspace trust prompts, and running the formalization pipeline in the background to avoid interruptions."
version: 1.0.1
---

# MathCode Agent

MathCode is a terminal-based AI coding assistant equipped with a built-in math formalization engine. It converts natural language math problems into **Lean 4** theorems and attempts formal proofs.

## Installation
```bash
git clone https://github.com/math-ai-org/mathcode.git ~/mathcode
cd ~/mathcode
bash setup.sh
```

## Configuration (Anthropic Backend)
By default, MathCode uses OpenAI Codex. If Codex CLI is not installed or you prefer Anthropic, configure the `~/mathcode/.env` file:
1. Set `MATHCODE_USE_OPENAI=0` (disables Codex OAuth).
2. Set `AUTOLEAN_USE_CODEX=0` (disables Codex backend for math tools).
3. Add `ANTHROPIC_API_KEY=sk-ant-...`

*Note: Environment variables exported in the shell will override the `.env` file.*

## Execution & Avoiding Interactive Prompts
MathCode's underlying engine will prompt for "workspace trust" if run interactively, which blocks autonomous execution. 
**To bypass interactive prompts:** Always use the `--dangerously-skip-permissions` flag combined with `-p`.

```bash
# Correct non-interactive execution
cd ~/mathcode
./run -p "Prove that the square of an even number is even" --dangerously-skip-permissions
```
*Note: The `-p` flag skips UI, but `--dangerously-skip-permissions` is required to bypass the workspace trust dialog entirely during tool execution.*

## Prompt Precision and Mathematical Ambiguity
MathCode is highly literal and sensitive to mathematical term overloading. If a requested lemma shares a name with a famous theorem in another domain (e.g., "Jordan's Lemma" in complex analysis vs. quantum computing), MathCode's evaluation engine may conflate them, resulting in a `.lean` file that attempts to conjoin unrelated domains and fails with a `sorry` placeholder.
**Always be highly specific** in your prompts:
- *Bad:* "Prove Jordan's Lemma (Grover's Algorithm)"
- *Good:* "Formalize and prove Jordan's Lemma for pairs of reflections on a finite-dimensional Hilbert space (as used in Grover's Algorithm)"

## Multi-Step Execution (Formalize then Prove)
Because Lean 4 formalization and proof generation are expensive loops, it often helps to separate them.
1. **Formalization:** MathCode will first attempt to translate the natural language problem into Lean 4 code. Math outputs and formalizations are saved to the `~/mathcode/LeanFormalizations/` directory as `.lean` and `.eval.json` files.
2. **Proving (WARNING: MASSIVE RUNTIME & CREDIT USAGE):** When a formalization is completed and saved with a `sorry` placeholder, do **not** try to run the underlying scripts directly. Instead, instruct the main `run` binary to complete the proof:
```bash
cd ~/mathcode
./run -p "Write the full Lean 4 proof for the theorem in LeanFormalizations/problem_problem.lean. Replace the 'sorry' with the actual proof using Mathlib tactics." --dangerously-skip-permissions
```
*Experiential Warning:* Generating heavy tactic proofs (like spectral theory or complex algebraic derivations) can take **50 to 60+ minutes** of non-stop iterative compiler loops. This will frequently hit hard execution timeouts in background cron workers, and it consumes massive amounts of Anthropic API credits. If the MathCode agent abruptly vanishes from the `ps aux` process list without updating the file, check your Anthropic API balance—it likely hit a "Credit balance is too low" error.

## Running Long Formalizations
MathCode MUST be run as a regular blocking (foreground) terminal call with a high `timeout` (e.g., 600s or 900s). Do **not** use `background=true` as it causes MathCode to silently hang or sleep indefinitely.

**CRITICAL:** Because it runs in the foreground, if the user sends a new message while the command is running, the system will interrupt it (Exit 130). Before starting a long MathCode command, you MUST explicitly warn the user: "Please do not send any messages while this is running, as it will interrupt the proof generation."

If you run them as a foreground blocking call, the command will be interrupted and killed (Exit Code 130) if the user happens to send a new message in the chat before it finishes.

**Important:** When running in the background, do **not** use terminal `background=true` as it causes MathCode to silently hang or sleep indefinitely. Instead, if the user requests running MathCode in the background while continuing a chat session, use the `cronjob` tool to dispatch a background agent to run the proof in an isolated session. The `delegate_task` tool is not recommended as it still blocks the current chat turn until the subagent completes.

**Cronjob Quirks:** When run via `cronjob`, MathCode's CLI will spam the cronjob logs with its `(｡◕‿◕｡)` loading spinner frames endlessly. This is normal and expected. To track actual progress, ignore the spinner output and instead check `ps aux | grep autolean` to see if the worker is active, and read the logs in `/var/folders/.../AUTOLEAN/logs/` (or wherever the temp working directory is created) using `search_files` and `read_file`.

```bash
# Good (Background)
./run -p "Prove Jordan's Lemma" --dangerously-skip-permissions

# Bad (Will fail in background)
echo "Prove Jordan's Lemma" | ./run -p --dangerously-skip-permissions
```