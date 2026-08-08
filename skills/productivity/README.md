# Productivity Skills

Workspace automation, document management, and productivity tool integrations for Hermes Agent.

## Overview

This category contains 8 skills for automating common productivity workflows — from managing tasks in Linear to editing presentations in PowerPoint. These skills integrate with popular tools like Google Workspace, Notion, Airtable, and more, helping you streamline repetitive work and build powerful automation pipelines.

## Available Skills

### Project & Task Management

#### **linear**
Manage Linear issues, projects, and teams via GraphQL API and curl.

**Use when:** Automating issue creation, project tracking, or building workflows around Linear.

**Key features:**
- Issue CRUD operations (create, read, update, delete)
- Project and team management
- GraphQL-based queries
- Custom workflow automation

---

#### **airtable**
Airtable REST API integration via curl — record management, filtering, and upserts.

**Use when:** Managing structured data in Airtable bases programmatically.

**Key features:**
- Record create, read, update, delete
- Advanced filtering and queries
- Upsert operations (update or insert)
- Batch operations support

---

#### **notion**
Notion API integration via curl — pages, databases, blocks, and search.

**Use when:** Automating Notion workspace management, content creation, or database operations.

**Key features:**
- Page and database management
- Block-level content manipulation
- Search and query operations
- Workspace automation

---

### Google Workspace Integration

#### **google-workspace**
Comprehensive Google Workspace automation — Gmail, Calendar, Drive, Docs, and Sheets via gws CLI or Python.

**Use when:** Automating email, calendar, document, or spreadsheet workflows across Google services.

**Key features:**
- Gmail automation (send, read, search, filter)
- Calendar event management
- Drive file operations
- Docs and Sheets manipulation
- Multi-service workflows

---

### Document Processing

#### **powerpoint**
Create, read, edit, and manipulate .pptx presentations — slides, notes, templates, and layouts.

**Use when:** Generating presentations programmatically, editing existing decks, or extracting content.

**Key features:**
- Slide creation and layout management
- Speaker notes and comments
- Template-based generation
- Content extraction and modification

---

#### **nano-pdf**
Edit PDF text, fix typos, and update titles using natural language prompts via nano-pdf CLI.

**Use when:** Making quick text edits to PDFs without manual tools.

**Key features:**
- Natural language editing commands
- Text modification and typo fixes
- Title and metadata updates
- LLM-powered PDF editing

---

#### **ocr-and-documents**
Extract text from PDFs and scanned documents using pymupdf and marker-pdf.

**Use when:** Converting scanned documents to text, extracting content from PDFs, or processing images with text.

**Key features:**
- OCR for scanned documents
- PDF text extraction (pymupdf)
- Advanced extraction with marker-pdf
- Multi-format document processing

---

### Location & Mapping

#### **maps**
Geocoding, points of interest, routing, and timezone data via OpenStreetMap and OSRM.

**Use when:** Converting addresses to coordinates, finding routes, discovering nearby places, or handling location data.

**Key features:**
- Address geocoding and reverse geocoding
- Point of interest (POI) discovery
- Route calculation and optimization (OSRM)
- Timezone lookup by coordinates

---

## Quick Start

### Example: Automate Issue Management

```bash
# Create a Linear issue from meeting notes
/linear "Create issue: Implement user authentication with OAuth2, assign to backend team, set priority to high"

# Update Notion database with status
/notion "Add task to Project Tracker: OAuth implementation - In Progress"
```

### Example: Document Workflow

```bash
# Extract text from scanned contract
/ocr-and-documents "Extract text from contract_scan.pdf"

# Edit specific sections
/nano-pdf "Change the date in section 2 from January to February"

# Generate presentation
/powerpoint "Create a deck with 5 slides about Q1 results: title, overview, metrics, challenges, next steps"
```

### Example: Google Workspace Automation

```bash
# Check calendar and send summary email
/google-workspace "Get today's calendar events and send summary to team@company.com"

# Create Drive folder structure and share
/google-workspace "Create folder Q1_Reports with subfolders Sales, Marketing, Engineering and share with engineering@company.com"
```

### Example: Location-Based Workflow

```bash
# Find nearby coffee shops
/maps "Find coffee shops within 1km of 123 Main St, San Francisco"

# Calculate delivery routes
/maps "Optimize route for deliveries to: [address1, address2, address3]"
```

## Skill Combinations

**Project Tracking Pipeline:**
1. Use `linear` to create and manage issues
2. Use `notion` to document decisions and requirements
3. Use `google-workspace` to schedule meetings and send updates

**Document Generation Workflow:**
1. Use `airtable` to pull data from your database
2. Use `powerpoint` to generate presentation
3. Use `google-workspace` to share via Drive and email

**Content Extraction Pipeline:**
1. Use `ocr-and-documents` to extract text from scans
2. Use `nano-pdf` to clean up and edit
3. Use `notion` or `google-workspace` to organize extracted content

**Meeting Coordination:**
1. Use `google-workspace` to check availability
2. Use `maps` to find meeting location and directions
3. Use `linear` or `notion` to create follow-up tasks

## Choosing the Right Tool

**For task management:**
- Agile workflows → `linear`
- Structured data → `airtable`
- Knowledge base → `notion`

**For documents:**
- PDFs (reading) → `ocr-and-documents`
- PDFs (editing) → `nano-pdf`
- Presentations → `powerpoint`
- All Google formats → `google-workspace`

**For data & automation:**
- Database-like → `airtable`
- Knowledge management → `notion`
- Email/calendar/files → `google-workspace`

## Common Workflows

### Daily Standup Automation
```bash
# 1. Pull completed tasks from Linear
/linear "List issues completed by me in the last 24 hours"

# 2. Get today's calendar
/google-workspace "Show my calendar for today"

# 3. Send standup summary
/google-workspace "Send standup summary email to team with completed tasks and today's schedule"
```

### Document Processing Pipeline
```bash
# 1. OCR scanned invoice
/ocr-and-documents "Extract text from invoice_scan.pdf"

# 2. Update amounts in existing PDF
/nano-pdf "Update total amount to $1,234.56 on page 1"

# 3. Store in Google Drive
/google-workspace "Upload processed invoice to Drive folder: Invoices/2026/April"
```

### Project Kickoff
```bash
# 1. Create project in Linear
/linear "Create project: Mobile App Redesign with epics for Design, Development, Testing"

# 2. Set up Notion workspace
/notion "Create project page with overview, timeline, team members, and decision log"

# 3. Schedule kickoff meeting
/google-workspace "Schedule 1-hour meeting tomorrow 2pm with design and dev teams, title: Mobile App Redesign Kickoff"
```

## Integration Tips

**API Keys:**
Most skills require API keys or authentication. Configure these using:
```bash
hermes config set
```

**Rate Limits:**
Be mindful of API rate limits when chaining multiple operations. Add delays if needed or batch operations where possible.

**Error Handling:**
Skills include built-in retry logic and error messages. Check logs if automation fails: `~/.hermes/logs/`

## Contributing

Found a bug or have an enhancement idea?

1. Open an issue describing the improvement
2. Fork the repository
3. Make changes to the relevant `SKILL.md`
4. Submit a pull request

## Related Categories

- **software-development/** - Development and coding workflows
- **github/** - GitHub integration
- **research/** - Research and analysis tools
- **creative/** - Design and content creation

---

**Questions?** Check the [Hermes Agent documentation](https://hermes-agent.nousresearch.com/docs/) or ask in the [Discord community](https://discord.gg/nousresearch).
