# Phonetic Captions — Architecture Diagrams

Visual reference for the end-to-end system built during the hackathon.

---

## 1. End-to-End System Flow

How a video goes from a Telegram message to a polished, captioned Short.

```mermaid
flowchart TD
    A([👤 User on Telegram\nsends video]) --> B

    subgraph Gateway["Hermes Gateway (gateway/run.py)"]
        B[Receive video file\ncache to disk]
        B --> C[Inject file path into\nagent message context]
    end

    C --> D

    subgraph Agent["Hermes Agent (run_agent.py)"]
        D[Pick up phonetic-captions skill\nroute → caption tool]
    end

    D --> E

    subgraph Pipeline["Caption Pipeline (tools/video_caption.py)"]
        E[🎙️ Whisper transcribe\nfaster-whisper medium\nVAD filter, auto-lang]
        E --> F[🤖 Kimi K2.5 classify\n1 LLM call: EN/VI classify\n+ diacritic correction\n+ phonetic guide]
        F --> G[📄 build_ass\nMAIN + PHONETIC styles\nASS subtitle file]
        G --> H[🎬 FFmpeg burn\nH.264, fast preset]
    end

    H --> I[💾 Save job JSON\n~/.hermes/caption-jobs/id.json]
    I --> J[Agent replies:\ncaptioned video\n+ dashboard link]
    J --> K([👤 User receives\nvideo on Telegram])

    J --> L

    subgraph Dashboard["Dashboard Visual Editor (plugins/phonetic-captions/)"]
        L[Open /captions/:id\nin browser]
        L --> M[Edit segments\ntext / phonetics / EN↔VI]
        M --> N[Change style\nfont, size, color, margins]
        N --> O[Re-burn\nFFmpeg only, no LLM]
        O --> P[Video player reloads]
    end

    P --> Q([👤 Download\nfinished Short])
```

---

## 2. Caption Pipeline — Internal Detail

What happens inside `tools/video_caption.py` for a single `caption` operation.

```mermaid
flowchart LR
    V[("📹 Input\nvideo")]

    subgraph Transcribe["Step 1 — Transcribe"]
        T1[faster-whisper\nmodel: medium\nlanguage: None\nVAD: silero]
        T2["Raw segments\n{id, start, end, text, lang}"]
        T1 --> T2
    end

    subgraph Classify["Step 2 — Classify & Phonetics"]
        K1["Kimi K2.5\n(NVIDIA NIM)\n1 API call, all segments"]
        K2["Enriched segments\n{lang: en|vi\ntext: corrected\nphonetic: [humm biet]}"]
        K1 --> K2
        NOTE["⚠️ Kimi quirk:\nresponse may land in\nreasoning_content\n→ fallback guard"]
    end

    subgraph ASS["Step 3 — Build ASS"]
        A1["build_ass()\nMAIN style: VI text bold\nPHONETIC style: italic smaller\n&H99... semi-transparent"]
        A2[".ass subtitle file"]
        A1 --> A2
    end

    subgraph Burn["Step 4 — Burn"]
        B1["FFmpeg\nsubtitles filter\nH.264 -preset fast"]
        B2[("📹 Captioned\nvideo")]
        B1 --> B2
    end

    V --> T1
    T2 --> K1
    K2 --> A1
    A2 --> B1
    K2 --> JOB[("💾 Job JSON\ncaption-jobs/id.json")]
    B2 --> JOB
```

---

## 3. Dashboard Plugin Architecture

How the plugin slots into the Hermes dashboard without touching core files.

```mermaid
flowchart TB
    subgraph Core["Hermes Core (unchanged)"]
        WS["web_server.py\nFastAPI app"]
        PL["Plugin loader\n/api/dashboard/plugins"]
        CATCH["App.tsx\n* catch-all\n(1-line guard added)"]
        WS --> PL
    end

    subgraph Plugin["plugins/phonetic-captions/dashboard/"]
        MAN["manifest.json\ntab: /captions\nicon: FileText\nafter: skills"]
        API["plugin_api.py\nAPIRouter()\n16 routes at\n/api/plugins/phonetic-captions/*"]
        UI["dist/index.js\npre-built 42.1kB IIFE\nReact + lucide-react\nno user build step"]
    end

    subgraph Store["Job Store"]
        JOBS[("~/.hermes/\ncaption-jobs/\n{id}.json")]
    end

    subgraph AI["AI Services"]
        AG["AIAgent\n(NL edits, QA,\nstyle suggestions)"]
        MEM["MemoryStore\n(style history\nacross sessions)"]
    end

    FFM["FFmpeg subprocess\n(re-burn)"]

    PL -->|discovers| MAN
    MAN -->|registers tab| WS
    MAN -->|mounts router| API
    WS -->|serves IIFE| UI
    UI <-->|16 REST endpoints| API
    API <--> JOBS
    API --> FFM
    FFM --> JOBS
    API <--> AG
    API --> MEM
    MEM --> AG
```

**Plugin API surface** (all at `/api/plugins/phonetic-captions/`):

| Endpoint | Purpose |
|---|---|
| `GET /jobs` | Job list with status badges |
| `GET /jobs/{id}` | Full job: segments + style + paths |
| `PUT /jobs/{id}/segments` | Save edited segments |
| `PUT /jobs/{id}/style` | Save style changes |
| `POST /jobs/{id}/burn` | Re-burn + write style diff to MemoryStore |
| `GET /jobs/{id}/video` | Stream video for browser player |
| `GET /jobs/{id}/download` | Download final output |
| `POST /upload` | Create job from upload; pipeline in background thread |
| `GET /jobs/{id}/status` | Poll status (`pending/transcribing/generating_phonetics/ready/error`) |
| `POST /jobs/{id}/nl-edit` | NL instruction → JSON patch array (propose only) |
| `POST /jobs/{id}/qa` | AI quality review → segment flag list |
| `GET /style/suggestion` | Cross-session style via MemoryStore + AIAgent |
| `GET /presets` | List all named style presets |
| `PUT /presets/{name}` | Save or overwrite a named preset |
| `DELETE /presets/{name}` | Delete a named preset |
| `POST /presets/generate` | NL description → AIAgent → CaptionStyle (not auto-saved) |

---

## 4. Interactive Edit Loop

The full cycle once the user is in the dashboard editor. No LLM tokens spent on re-burn — only on NL edits, QA, and style suggestions.

```mermaid
sequenceDiagram
    actor User
    participant UI as Dashboard UI<br/>(React)
    participant API as Plugin API<br/>(plugin_api.py)
    participant Store as Job JSON<br/>(~/.hermes/caption-jobs/)
    participant FFmpeg
    participant Agent as AIAgent
    participant Mem as MemoryStore

    Note over User,Mem: Manual edit cycle (no LLM)
    User->>UI: Edit text / phonetic / EN↔VI badge
    User->>UI: Adjust style panel
    User->>UI: Click Re-burn
    UI->>API: PUT /jobs/{id}/segments
    UI->>API: PUT /jobs/{id}/style
    UI->>API: POST /jobs/{id}/burn
    API->>Store: Write updated job JSON
    API->>FFmpeg: Burn with new ASS file
    FFmpeg-->>API: Exit 0
    API->>Mem: Append style diff (silent)
    API-->>UI: { status: ready }
    UI->>UI: Reload video player

    Note over User,Mem: Natural-language edit (LLM propose only)
    User->>UI: Type instruction\n"shift segment 4 forward 0.3s"
    UI->>API: POST /jobs/{id}/nl-edit
    API->>Agent: Instruction + current segments
    Agent-->>API: JSON patch array
    API-->>UI: Diff with before/after
    UI->>User: Show per-change checkboxes
    User->>UI: Approve selected patches
    UI->>API: PUT /jobs/{id}/segments (approved only)

    Note over User,Mem: QA review
    User->>UI: Click "Review all"
    UI->>API: POST /jobs/{id}/qa
    API->>Agent: All segments for review
    Agent-->>API: Flag list {id, issue, suggestion}
    API-->>UI: Flagged segment IDs
    UI->>User: Amber borders on problem segments
    User->>UI: Click "Fix with AI" on segment
    UI->>UI: Pre-fill NL panel with suggestion

    Note over User,Mem: Style suggestion (≥3 burns)
    User->>UI: Hermes panel → Style presets → Learned card
    UI->>API: GET /style/suggestion
    API->>Mem: Read style diff history
    API->>Agent: History + request analysis
    Agent-->>API: CaptionStyle + explanation
    API-->>UI: Suggested style JSON
    UI->>User: Amber Learned card (Apply / Save as)

    Note over User,Mem: Named preset library
    User->>UI: Click "Save current style", enter name
    UI->>API: PUT /presets/{name}
    API-->>UI: { ok: true }
    UI->>User: Card added to preset gallery
    User->>UI: Click preset card
    UI->>UI: Apply style to local state
    User->>UI: "Create with AI" → describe style
    UI->>API: POST /presets/generate
    API->>Agent: NL description
    Agent-->>API: CaptionStyle JSON
    API-->>UI: Style preview
    UI->>User: Preview card with name input + Save/Apply
```

---

## 5. On-Screen Caption Layout

What the burned video actually looks like for each segment type.

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│                  [ video content ]                      │
│                                                         │
│                                                         │
│                                                         │
│   Vietnamese segment:                                   │
│                                                         │
│              không biết                                 │
│            ─────────────                                │
│             [humm biet]                                 │
│                                                         │
│   ──────────────────────────────────────────────────    │
│                                                         │
│   English segment:                                      │
│                                                         │
│           Today we learn how to say...                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

  MAIN style:      bold, white, black outline, bottom-center
  PHONETIC style:  italic, smaller (~80% size), semi-transparent
                   only rendered for lang=vi segments
```

**ASS style params** (defaults from `caption.style` in config.yaml):

| Style | Font | Size | Color | Outline | Alignment |
|---|---|---|---|---|---|
| MAIN | Arial | 48 | White `&H00FFFFFF` | Black 3px | 2 (bottom-center) |
| PHONETIC | Arial | 38 | White `&H99FFFFFF` | Black 2px | 8 (top-center of MAIN) |

---

## 6. Segment Data Model

The shape of data through the pipeline and stored in the job JSON.

```mermaid
erDiagram
    JOB {
        string id
        string status
        string video_path
        string output_path
        string ass_path
        datetime created_at
        CaptionStyle style
    }

    SEGMENT {
        int id
        float start
        float end
        string text
        string lang
        string phonetic
    }

    CAPTION_STYLE {
        string font
        int font_size
        string primary_color
        string outline_color
        int outline_width
        int alignment
        int margin_bottom
        int max_line_length
    }

    JOB ||--o{ SEGMENT : "contains"
    JOB ||--|| CAPTION_STYLE : "has"
```

**`lang` values**: `"en"` — English text only (MAIN style); `"vi"` — Vietnamese text + phonetic guide (MAIN + PHONETIC styles)

**`phonetic` field**: Only populated when `lang = "vi"`. Format: `[space-separated English approximation]` e.g. `[humm biet]` for `không biết`. Empty string for `lang = "en"` segments.

---

## 7. Component Dependency Map

Where each piece lives in the repo and how they relate.

```mermaid
flowchart LR
    subgraph Tools["tools/"]
        VC["video_caption.py\n(tool implementation)"]
    end

    subgraph Toolsets["toolsets.py"]
        TS["video-caption\ntoolset entry"]
    end

    subgraph Skills["skills/video/phonetic-captions/"]
        SK["SKILL.md\n(prompt + instructions\nfor agent)"]
    end

    subgraph Gateway["gateway/run.py"]
        GW["_preprocess_inbound_text()\nvideo MIME injection"]
    end

    subgraph Config["hermes_cli/config.py"]
        CF["DEFAULT_CONFIG\ncaption.style section"]
    end

    subgraph Plugin["plugins/phonetic-captions/dashboard/"]
        MN["manifest.json"]
        PA["plugin_api.py"]
        FE["dist/index.js"]
        SRC["src/index.tsx"]
        BLD["build.mjs (esbuild)"]
        SRC -->|builds to| FE
        BLD --> FE
    end

    VC --> TS
    CF --> VC
    SK --> VC
    GW -->|path injected into message| SK
    MN --> PA
    PA --> VC
```
