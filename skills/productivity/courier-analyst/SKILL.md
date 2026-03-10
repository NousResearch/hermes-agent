---
name: courier-analyst
description: Intelligent file analysis with subagent delegation and artifact delivery. Analyzes uploaded files (CSV, logs, code, documents), produces structured reports with insights, and sends results back to the user via send_file.
version: 1.0.0
author: eren-karakus0
license: MIT
metadata:
  hermes:
    tags: [Analysis, Files, Delegation, Artifacts, Productivity]
    platforms: [cli, telegram, discord]
---

# Courier Analyst

Analyze uploaded files using parallel subagent delegation, then deliver structured reports and artifacts back to the user.

## When to Use

Activate this skill when:
- A user sends a file (CSV, log, JSON, code archive, document) and asks for analysis
- The user wants insights, statistics, patterns, or quality checks on data
- The user needs a structured report generated from raw data
- Multiple analysis perspectives would add value (statistics + quality review)

## Required Toolsets

This skill requires: `terminal`, `file`, `file_transfer`, `delegation`

## Workflow

### Step 1: File Detection & Strategy

Identify the file type and choose the analysis approach:

| File Type | Strategy | Tools |
|-----------|----------|-------|
| CSV/Excel | Statistical analysis + data quality | pandas, matplotlib |
| Log files | Error pattern detection + anomaly timeline | grep, regex, awk |
| JSON/YAML | Schema analysis + validation | jq, python json |
| Code (.py, .js, .zip) | Structure analysis + review | ast, eslint, tree |
| PDF/Docs | Content extraction + summary | text extraction |
| Generic | Content overview + size/encoding info | file, wc, head |

### Step 2: Parallel Analysis via Subagents

Use `delegate_task` in batch mode with 2 parallel subagents:

**Subagent 1 — Analyst:**
```
Analyze the file at {path}. Produce a structured report with:
- File overview (type, size, encoding, structure)
- Key statistics and metrics
- Notable patterns or trends
- Data quality observations
- Summary of findings

Save the report as /workspace/reports/{name}_analysis.md
If the data supports visualization, generate a chart using matplotlib
and save it as /workspace/reports/{name}_chart.png
```

**Subagent 2 — Reviewer:**
```
Review the file at {path} for quality and anomalies:
- Check for missing values, duplicates, outliers
- Identify encoding issues or malformed entries
- Flag potential data integrity concerns
- Assess completeness and consistency

Save findings as /workspace/reports/{name}_review.md
```

### Step 3: Merge & Deliver

1. Read both subagent reports from `/workspace/reports/`
2. Merge into a unified analysis document
3. Use `send_file` to deliver:
   - The merged analysis report (markdown)
   - Any generated charts or visualizations
   - The original subagent reports (if detailed)

## Analysis Templates

### CSV Analysis Report Structure
```markdown
# Data Analysis Report: {filename}

## Overview
- Rows: {count} | Columns: {count}
- File size: {size} | Encoding: {encoding}

## Column Summary
| Column | Type | Non-null | Unique | Sample Values |
|--------|------|----------|--------|---------------|
| ...    | ...  | ...      | ...    | ...           |

## Key Findings
1. {finding_1}
2. {finding_2}

## Data Quality
- Missing values: {count} ({percentage}%)
- Duplicates: {count}
- Anomalies: {list}

## Recommendations
- {recommendation_1}
```

### Log Analysis Report Structure
```markdown
# Log Analysis Report: {filename}

## Overview
- Total lines: {count} | Time range: {start} - {end}
- Log levels: ERROR({n}), WARN({n}), INFO({n})

## Error Patterns
| Pattern | Count | First Seen | Last Seen |
|---------|-------|------------|-----------|
| ...     | ...   | ...        | ...       |

## Timeline
- {timestamp}: {event}

## Anomalies
- {anomaly_description}
```

## Example Interaction

**User sends:** `sales_data.csv` with message "analyze this"

**Agent workflow:**
1. Detect: CSV file, 5MB, 50K rows
2. Delegate to 2 subagents (analyst + reviewer)
3. Analyst produces statistical report + bar chart
4. Reviewer flags 3% missing values in 'region' column
5. Merge reports into unified analysis
6. `send_file` delivers: analysis.md + chart.png

## Tips

- For large files (>10MB), sample first 10K rows for quick analysis, then full scan
- Always create the `/workspace/reports/` directory before saving
- Use `send_file` for each artifact separately (one file per call)
- If matplotlib is not available, install it: `pip install matplotlib`
- For code archives, extract first: `unzip code.zip -d /workspace/code_review/`
