# Research Skills

Academic research workflows, paper discovery, knowledge management, and publication tools for Hermes Agent.

## Overview

This category contains 5 skills for research workflows — from discovering papers on arXiv to writing publication-ready manuscripts for top ML conferences. Whether you're conducting literature reviews, monitoring the latest research, building knowledge bases, or writing your own papers, these skills provide professional research operations.

## Available Skills

### Paper Discovery & Search

#### **arxiv**
Search arXiv papers by keyword, author, category, or paper ID.

**Use when:** Finding relevant research papers, tracking specific authors, or exploring categories.

**Key features:**
- Keyword search across titles, abstracts, and full text
- Author-based queries
- Category filtering (cs.AI, cs.LG, etc.)
- Direct paper ID lookup
- Date range filtering
- Sorting by relevance or recency

---

### Information Monitoring

#### **blogwatcher**
Monitor blogs and RSS/Atom feeds using the blogwatcher-cli tool.

**Use when:** Tracking research blogs, staying updated on specific topics, or monitoring industry news.

**Key features:**
- RSS/Atom feed monitoring
- Blog update tracking
- Keyword filtering
- Aggregated feed reading
- Custom monitoring workflows

---

### Knowledge Management

#### **llm-wiki** (Karpathy's LLM Wiki)
Build and query interlinked markdown knowledge bases.

**Use when:** Creating personal research wikis, organizing interconnected notes, or building knowledge graphs.

**Key features:**
- Markdown-based knowledge base
- Interlinked notes and references
- Query and search functionality
- Graph-like knowledge organization
- Based on Karpathy's LLM wiki approach

---

### Prediction Markets & Research Trends

#### **polymarket**
Query Polymarket prediction markets — prices, orderbooks, market history.

**Use when:** Researching market predictions, analyzing crowd wisdom, or tracking real-world events.

**Key features:**
- Market search and discovery
- Current price queries
- Orderbook inspection
- Historical price data
- Event outcome tracking
- Probability analysis

**Use cases:**
- Research on prediction accuracy
- Crowd wisdom analysis
- Event probability tracking
- Market mechanism research

---

### Academic Writing

#### **research-paper-writing**
Write machine learning papers for NeurIPS, ICML, ICLR — from design to submission.

**Use when:** Writing academic papers for top-tier ML conferences.

**Key features:**
- Conference-specific LaTeX templates (NeurIPS, ICML, ICLR)
- Paper structure guidance
- Section templates (abstract, intro, methods, experiments, conclusion)
- Citation management
- Figure and table formatting
- Submission preparation
- Complete design-to-submit workflow

**Supported conferences:**
- NeurIPS (Conference on Neural Information Processing Systems)
- ICML (International Conference on Machine Learning)
- ICLR (International Conference on Learning Representations)

---

## Quick Start

### Example: Literature Review

```bash
# 1. Search for recent papers
/arxiv "Search for papers on 'transformer attention mechanisms' from 2024"

# 2. Monitor key researchers
/blogwatcher "Monitor Andrej Karpathy's blog and OpenAI research updates"

# 3. Build knowledge base
/llm-wiki "Create knowledge graph linking transformer papers and key concepts"
```

### Example: Research Monitoring

```bash
# 1. Track specific categories
/arxiv "Show latest papers in cs.LG and cs.AI from the past week"

# 2. Monitor research blogs
/blogwatcher "Add feeds: Distill.pub, Google AI Blog, DeepMind blog"

# 3. Check prediction markets
/polymarket "What's the probability of GPT-5 release in 2025?"
```

### Example: Writing a Paper

```bash
# 1. Set up paper structure
/research-paper-writing "Create NeurIPS 2026 paper template for 'Efficient Attention Mechanisms'"

# 2. Continue literature review
/arxiv "Find related work on efficient transformers"

# 3. Build references in knowledge base
/llm-wiki "Add key papers to related work knowledge graph"
```

## Skill Combinations

**Complete Research Workflow:**
1. Use `arxiv` to discover relevant papers
2. Use `blogwatcher` to monitor latest developments
3. Use `llm-wiki` to organize findings in knowledge base
4. Use `research-paper-writing` to write up results

**Literature Review Pipeline:**
1. `arxiv` — Search papers by topic, author, or category
2. `llm-wiki` — Build interconnected notes linking papers
3. `research-paper-writing` — Write related work section

**Research Monitoring:**
1. `arxiv` — Daily paper tracking by category
2. `blogwatcher` — Monitor key research blogs
3. `polymarket` — Track predictions related to research directions

**Paper Writing Process:**
1. `llm-wiki` — Organize notes and ideas
2. `arxiv` — Comprehensive literature review
3. `research-paper-writing` — Draft, refine, submit

## Choosing the Right Tool

**For finding papers:**
- Academic papers → `arxiv`
- Blog posts & articles → `blogwatcher`

**For organizing research:**
- Interconnected notes → `llm-wiki`
- Flat file system → Use other note-taking tools

**For writing:**
- ML conference papers → `research-paper-writing`
- Other formats → Use general writing tools

**For tracking trends:**
- Research blogs → `blogwatcher`
- Prediction markets → `polymarket`
- Academic papers → `arxiv`

## Common Workflows

### Daily Research Routine

```bash
# Morning: Check new papers
/arxiv "Show papers from cs.LG submitted yesterday, sorted by relevance"

# Midday: Check research blogs
/blogwatcher "Check updates from monitored feeds"

# Evening: Update knowledge base
/llm-wiki "Add today's interesting papers with key insights"
```

### Writing a Conference Paper

```bash
# Week 1: Literature review
/arxiv "Comprehensive search on [your topic]"
/llm-wiki "Build related work knowledge graph"

# Week 2-4: Experiments and writing
/research-paper-writing "Create ICML template, write methods section"

# Week 5: Polish and refine
/arxiv "Find any missing recent papers"
/research-paper-writing "Finalize references, check formatting"

# Week 6: Submit
/research-paper-writing "Prepare camera-ready submission for ICML"
```

### Research Topic Exploration

```bash
# 1. Find seminal papers
/arxiv "Search 'attention is all you need' and related citations"

# 2. Track recent developments
/arxiv "Papers on transformer improvements from past 6 months"

# 3. Monitor expert opinions
/blogwatcher "Add feeds from transformer researchers"

# 4. Check market predictions
/polymarket "Will transformers be replaced by new architecture in 2026?"

# 5. Organize findings
/llm-wiki "Create transformer evolution timeline and concept map"
```

## Research Best Practices

**Paper Discovery:**
- Use specific keywords and categories
- Track key authors in your field
- Set up regular arXiv alerts
- Cross-reference with citations

**Knowledge Management:**
- Link related papers in your wiki
- Summarize key contributions
- Track open questions
- Note potential collaborations

**Paper Writing:**
- Start with outline and structure
- Write methods before results
- Iterate on abstract and intro
- Get feedback early and often
- Follow conference guidelines strictly

**Staying Current:**
- Daily arXiv check in your categories
- Weekly blog roundup
- Monthly deep dives into key papers
- Track prediction markets for field trends

## Conference Submission Tips

**Timeline (typical):**
- 8 weeks: Initial experiments and outline
- 6 weeks: Methods and preliminary results
- 4 weeks: Full draft with all experiments
- 2 weeks: Internal review and revision
- 1 week: Final polish and submission prep
- Submission: On-time, well-formatted

**Key Sections:**
- **Abstract:** Clear problem, method, results (250 words)
- **Intro:** Motivation, contributions, roadmap
- **Related Work:** Comprehensive, positions your work
- **Methods:** Reproducible, detailed
- **Experiments:** Thorough, ablations, baselines
- **Conclusion:** Summary, limitations, future work

**LaTeX Tips:**
- Use conference templates from `research-paper-writing` skill
- Keep figures vector format (PDF) when possible
- Number all sections, figures, tables
- Use consistent notation throughout
- Proofread multiple times

## Integration with Other Categories

**Combine with MLOps:**
- Use `arxiv` to find papers on techniques
- Implement using `mlops/` skills
- Write up results with `research-paper-writing`

**Combine with Creative:**
- Generate figures with `creative/` skills
- Create diagrams with `architecture-diagram`
- Design infographics for presentations

**Combine with Software Development:**
- Implement paper methods
- Create reproducible code
- Document with proper citations

## Contributing

Found a bug or have an enhancement idea?

1. Open an issue describing the improvement
2. Fork the repository
3. Make changes to the relevant `SKILL.md`
4. Submit a pull request

## Related Categories

- **mlops/** - ML experiment tracking and evaluation
- **creative/** - Figure and diagram creation
- **software-development/** - Code implementation
- **productivity/** - Note-taking and document management

---

**Questions?** Check the [Hermes Agent documentation](https://hermes-agent.nousresearch.com/docs/) or ask in the [Discord community](https://discord.gg/nousresearch).
