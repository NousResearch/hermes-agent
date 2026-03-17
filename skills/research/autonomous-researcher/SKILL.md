---
name: autonomous-researcher
description: Complete autonomous ML research workflow. Orchestrates literature review, ideation, implementation, data preparation, training, evaluation, iteration, and publication. Routes to specialized skills for each phase. Use for end-to-end research projects.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [research, autonomous, ml, training, evaluation, papers]
    homepage: https://github.com/NousResearch/hermes-agent
    requires:
      - arxiv
      - duckduckgo-search
      - codex
      - claude-code
      - hermes-agent-spawning
      - tinker
      - evaluating-llms-harness
      - ml-paper-writing
      - github-repo-management
---

# Autonomous Researcher

End-to-end autonomous ML research workflow. This skill orchestrates all research phases and routes to specialized skills for each step.

## Research Pipeline

\`\`\`
1. LITERATURE    -->  2. IDEATION    -->  3. IMPLEMENTATION
   REVIEW              HYPOTHESIS           CODE + DATA
   (arxiv)             (synthesis)          (codex/claude-code)

4. TRAINING      -->  5. EVALUATION  -->  6. ITERATION
   (tinker)            (lm-eval)           COMPARE + REFINE

7. WRITING       -->  8. PUBLISHING
   (ml-paper)          (HF Hub, GitHub)
\`\`\`

## Input Modes

The Autonomous Researcher can start from different points depending on what the user provides:

### Mode A: Full Autonomous (Start from Scratch)
User provides: Nothing or just a topic
Pipeline: Literature Review -> Ideation -> Implementation -> Training -> Evaluation -> Iteration -> Writing -> Publishing

### Mode B: User-Provided Hypothesis
User provides: Problem statement, research idea/hypothesis
Pipeline: Skip literature review, validate hypothesis -> Implementation -> Training -> Evaluation -> Iteration -> Writing -> Publishing

### Mode C: User-Provided Data
User provides: Dataset + (optional) hypothesis
Pipeline: Data analysis -> (hypothesis or ideation) -> Implementation -> Training -> Evaluation -> Iteration -> Writing -> Publishing

### Mode D: User-Provided Everything
User provides: Hypothesis + Data + Execution preferences
Pipeline: Direct to implementation with user's specifications

### Mode E: Continue Existing Project
User provides: Checkpoint/existing work
Pipeline: Resume from checkpoint -> Continue training -> Evaluation -> Iteration -> Writing -> Publishing

---

## Phase 0: Project Initialization

Before starting any research, set up the project structure:

\`\`\`python
def initialize_project(user_input=None):
    # Create directory structure
    dirs = [
        "~/research/project/src",
        "~/research/project/configs",
        "~/research/project/data",
        "~/research/project/logs",
        "~/research/project/results",
        "~/research/project/checkpoints",
        "~/research/project/paper",
        "~/research/literature",
        "~/research/notes",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Determine mode based on user input
    if user_input:
        mode = determine_mode(user_input)
    else:
        mode = "full_autonomous"
    
    # Write project config
    config = {
        "mode": mode,
        "start_time": datetime.now().isoformat(),
        "user_input": user_input,
        "status": "initialized",
        "phase": "start",
    }
    write_file("~/research/project/config.yaml", yaml.dump(config))
    
    return mode
\`\`\`

---

## Phase 1: Literature Review

### Primary Tools
- \`arxiv\` skill - Search and retrieve academic papers
- \`web_search\` - Find blog posts, documentation, discussions
- \`web_extract\` - Extract content from URLs
- \`duckduckgo-search\` - Fallback search

### When to Skip
- Mode B: User provided hypothesis -> Skip or do minimal review
- Mode C: User provided data only -> Do targeted review
- Mode D: User provided everything -> Skip entirely
- Mode E: Continue existing -> Skip

### Workflow

\`\`\`python
# 1. Search arXiv for relevant papers
papers = web_search("grpo reinforcement learning fine-tuning", limit=20)

# 2. Extract key papers
for paper in papers[:5]:
    content = web_extract([paper.url])

# 3. Search for implementations
implementations = web_search(f"{topic} github implementation")

# 4. Identify gaps
# - What problems remain unsolved?
# - What baselines are used?
# - What datasets are common?
\`\`\`

### Output Artifacts
- \`~/research/literature_review.md\` - Summary of papers
- \`~/research/papers/\` - Downloaded PDFs
- \`~/research/gaps.md\` - Identified research gaps
- \`~/research/baselines.md\` - Common baselines and metrics

---

## Phase 2: Ideation & Hypothesis

### Workflow

\`\`\`python
# Spawn multiple agents to explore different angles
terminal(command="hermes chat -q 'Analyze literature review and propose 3 research directions. Write to ~/research/proposals.md'", background=True)

# Synthesize findings
# Combine literature insights with proposed directions
# Identify most promising approach
\`\`\`

### Decision Points
1. Is the hypothesis novel?
2. Is it feasible with available compute?
3. Is there existing work that already solved this?
4. What's the expected impact if successful?

### Output Artifacts
- \`~/research/hypothesis.md\` - Clear problem statement
- \`~/research/approach.md\` - Proposed methodology
- \`~/research/expected_results.md\` - Anticipated outcomes

---

## Phase 3: Implementation

### Primary Tools
- \`codex\` skill - OpenAI Codex for implementation
- \`claude-code\` skill - Claude Code for complex development
- \`hermes-agent-spawning\` - Parallel development

### Workflow

\`\`\`python
# 1. Create project structure
terminal(command="mkdir -p ~/research/project/{src,data,configs,scripts,logs}")

# 2. Implement base model/training code using Claude Code
terminal(command="""
hermes chat -q '
Create a training script for GRPO fine-tuning with the following:
- Load model from HuggingFace
- Implement GRPO loss function
- Support checkpoint saving/resuming
Write to ~/research/project/src/train.py
'
""", background=True)

# 3. Implement data processing
terminal(command="""
hermes chat -q '
Create data loading and preprocessing for the dataset.
Write to ~/research/project/src/data.py
'
""", background=True)
\`\`\`

### Code Structure
\`\`\`
project/
├── src/
│   ├── train.py          # Main training loop
│   ├── data.py           # Data loading/preprocessing
│   ├── model.py          # Model architecture
│   ├── loss.py           # Loss functions
│   └── utils.py          # Utilities
├── configs/
│   ├── base.yaml         # Default config
│   └── experiment.yaml   # Experiment-specific
├── scripts/
│   ├── run.sh            # Training launcher
│   └── eval.sh           # Evaluation launcher
├── data/
│   └── (downloaded datasets)
└── logs/
    └── (training logs)
\`\`\`

---

## Phase 4: Training

### Primary Tools
- \`tinker\` skill - GPU training via API
- \`hermes-agent-spawning\` - Background training jobs

### Workflow (Using Tinker)

\`\`\`python
import tinker
from tinker import types

# 1. Initialize training
service = tinker.ServiceClient()
trainer = service.create_lora_training_client(
    base_model="meta-llama/Llama-3.1-8B",
    rank=32
)

# 2. Training loop (with checkpointing)
for step in range(total_steps):
    trainer.forward_backward(data, loss_fn="cross_entropy").result()
    trainer.optim_step(tinker.AdamParams(lr=1e-4)).result()
    
    if step % 50 == 0:
        # Checkpoint
        checkpoint = trainer.save_state(f"step_{step}").result().path
        save_training_state(step, checkpoint)
        
        # Evaluate
        sampler = trainer.save_weights_and_get_sampling_client(f"eval_{step}")
        metrics = evaluate(sampler, val_data)
\`\`\`

### Autonomous Training Pattern

For long training runs that span sessions, use the self-chaining pattern from the Tinker skill:

\`\`\`python
# autonomous_train.py
def main():
    state = load_state()  # Resume from checkpoint
    trainer = setup_training(state["checkpoint"])
    
    for step in range(state["step"], CHUNK_SIZE):
        train_step(trainer)
        
        if step % CHECKPOINT_EVERY == 0:
            checkpoint = trainer.save_state(f"step_{step}").result().path
            save_state({"step": step, "checkpoint": checkpoint})
    
    metrics = evaluate(trainer)
    decision = should_continue(metrics)
    
    if decision == "continue":
        spawn_next_chunk()
    else:
        save_final_results()

# Start autonomous loop
terminal(command="hermes chat -q 'Run autonomous_train.py'", background=True)
\`\`\`

### Parallel Experiments

\`\`\`python
# Run multiple experiments in parallel
terminal(command="hermes chat -q 'Run experiment A: GRPO with lr=1e-4'", background=True)
terminal(command="hermes chat -q 'Run experiment B: GRPO with lr=5e-5'", background=True)
terminal(command="hermes chat -q 'Run experiment C: DPO baseline'", background=True)
\`\`\`

---

## Phase 5: Evaluation

### Primary Tools
- \`evaluating-llms-harness\` skill - Standard benchmarks
- Custom evaluation scripts
- Tinker sampling client

### Workflow

\`\`\`python
# 1. Download checkpoint
tinker checkpoint download tinker://model_id/final

# 2. Run standard benchmarks
python -m lm_eval --model hf \\
    --model_args pretrained=./checkpoint \\
    --tasks mmlu,gsm8k,hellaswag,truthfulqa \\
    --batch_size 8

# 3. Compare with baselines
baseline_results = load_baseline_results()
compare_results(current_results, baseline_results)
\`\`\`

### Output Artifacts
- \`~/research/project/results/eval_results.json\` - Benchmark results
- \`~/research/project/results/comparison.md\` - Baseline comparison
- \`~/research/project/results/samples/\` - Generated samples for qualitative review

---

## Phase 6: Iteration

### Workflow

\`\`\`python
# 1. Analyze results
results = load_results()

# 2. Identify issues
issues = []
if results["mmlu"] < baseline["mmlu"]:
    issues.append("MMLU degraded - check for catastrophic forgetting")

# 3. Propose fixes
for issue in issues:
    fix = propose_fix(issue)
    implement_fix(fix)

# 4. Re-run training
terminal(command="hermes chat -q 'Run training iteration 2 with fixes'", background=True)
\`\`\`

### Iteration Decision Tree

\`\`\`
Results worse than baseline?
├── Yes -> Check for bugs
│         ├── Training converged? -> Try different hyperparameters
│         ├── Data issues? -> Check data quality/preprocessing
│         └── Implementation issues? -> Debug code
│
└── No -> Results better than baseline?
          ├── Yes, significantly -> Proceed to publication
          ├── Yes, marginally -> Run more experiments to confirm
          └── No change -> Try different approach
\`\`\`

---

## Phase 7: Paper Writing

### Primary Tools
- \`ml-paper-writing\` skill - Publication-ready papers
- LaTeX compilation

### Workflow

\`\`\`python
# 1. Generate paper outline using ml-paper-writing skill
# 2. Write sections
sections = {
    "abstract": "Summarize contributions, method, results",
    "introduction": "Problem, motivation, contributions",
    "related_work": "Literature review summary",
    "method": "Detailed approach description",
    "experiments": "Setup, datasets, baselines, results",
    "analysis": "Ablations, error analysis",
    "conclusion": "Summary, limitations, future work"
}

# 3. Generate figures
# Training curves, comparison tables, ablation charts

# 4. Compile LaTeX
terminal(command="cd ~/research/project/paper && pdflatex main.tex")
\`\`\`

---

## Phase 8: Publishing

### Primary Tools
- HuggingFace Hub (model upload)
- \`github-repo-management\` skill (code release)

### Workflow

\`\`\`python
# 1. Upload model to HuggingFace Hub
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./checkpoint",
    repo_id="username/model-name",
    repo_type="model"
)

# 2. Create model card with training details and results

# 3. Push code to GitHub
terminal(command="git add . && git commit -m 'Release model and code' && git push")

# 4. Create release
gh release create v1.0 --title "Initial Release"
\`\`\`

---

## Skill Routing Table

| Phase | Primary Skill | Secondary Skills |
|-------|--------------|------------------|
| Literature Review | \`arxiv\` | \`duckduckgo-search\`, \`web_extract\` |
| Ideation | \`hermes-agent-spawning\` | - |
| Implementation | \`codex\`, \`claude-code\` | \`hermes-agent-spawning\` |
| Training | \`tinker\` | \`axolotl\`, \`fine-tuning-with-trl\` |
| Evaluation | \`evaluating-llms-harness\` | Custom scripts |
| Iteration | \`tinker\` | \`evaluating-llms-harness\` |
| Writing | \`ml-paper-writing\` | - |
| Publishing | \`github-repo-management\` | HuggingFace Hub |

---

## Example Invocations by Mode

### Mode A: Full Autonomous Research

\`\`\`
User: "Research methods for improving LLM reasoning through reinforcement learning"

Hermes: *Starts complete pipeline*
1. Searches arXiv for RL + LLM papers
2. Identifies gaps in current research
3. Proposes novel approach (e.g., GRPO with curriculum learning)
4. Implements training code
5. Runs experiments on Tinker
6. Evaluates results
7. Writes paper
8. Publishes to HuggingFace/GitHub
\`\`\`

### Mode B: User-Provided Hypothesis

\`\`\`
User: "I want to test if curriculum learning improves GRPO fine-tuning.
The hypothesis is that starting with easy examples and progressively
increasing difficulty will improve sample efficiency by 20%."

Hermes: *Validates hypothesis and proceeds*
1. Quick literature check for prior curriculum learning + GRPO work
2. Confirms novelty
3. Creates experiment plan with baseline and treatment
4. Implements both variants
5. Runs A/B comparison
6. Reports results
\`\`\`

### Mode C: User-Provided Data

\`\`\`
User: "I have a dataset of math problems at ~/data/math_problems.jsonl.
Fine-tune a model to improve on GSM8K using this data."

Hermes: *Analyzes data and proceeds*
1. Analyzes dataset: 50K problems, solution steps average 150 tokens
2. Checks data quality
3. Proposes approach: SFT on solution steps, then GRPO for reasoning
4. Creates train/val split
5. Implements data pipeline
6. Runs training
7. Evaluates on GSM8K
\`\`\`

### Mode E: Continue Existing Project

\`\`\`
User: "Continue training from checkpoint tinker://model_abc/step_5000.
The model was training on GSM8K with GRPO. Evaluate current results
and continue if loss is still decreasing."

Hermes: *Resumes from checkpoint*
1. Loads checkpoint
2. Evaluates current model
3. Continues training if warranted
4. Stops when converged
5. Saves final checkpoint and writes analysis
\`\`\`

---

## Quick Reference Commands

### Start New Research Project

\`\`\`bash
hermes chat -q '
Start a new autonomous research project on [TOPIC].
1. Search arXiv for recent papers
2. Identify research gaps
3. Propose a novel approach
4. Create project structure
5. Begin implementation
'
\`\`\`

### Run Training with Monitoring

\`\`\`bash
hermes chat -q '
Start autonomous training on Tinker for the [PROJECT] model.
Use checkpoint-based resumption and evaluate every 100 steps.
'
\`\`\`

### Evaluate and Compare

\`\`\`bash
hermes chat -q '
Evaluate the trained model at ~/research/project/checkpoint.
Run benchmarks: MMLU, GSM8K, Hellaswag, TruthfulQA.
Compare with baseline results.
'
\`\`\`

### Write and Publish Paper

\`\`\`bash
hermes chat -q '
Write a NeurIPS-format paper based on the research in ~/research/project.
Include all sections and generate figures from results.
'
\`\`\`

---

## Troubleshooting

**Training won't start**
- Verify TINKER_API_KEY is set
- Check data format matches Tinker expectations
- Ensure checkpoint path is valid if resuming

**Evaluation fails**
- Check model checkpoint downloaded correctly
- Verify lm-evaluation-harness installed
- Ensure benchmark names match expected format

**Paper writing stalls**
- Ensure all experiments completed
- Check results files are valid JSON
- Run LaTeX compilation manually to debug

**Publishing errors**
- Verify HuggingFace credentials configured
- Check GitHub authentication
- Ensure model card is complete
HERMES_EOF; __hermes_rc=$?; printf '__HERMES_FENCE_a9f7b3__'; exit $__hermes_rc