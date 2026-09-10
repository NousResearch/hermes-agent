# Creative Skills

Design, art generation, multimedia creation, and visual communication tools for Hermes Agent — from architecture diagrams to AI music.

## Overview

This category contains 16 skills for creative work spanning visual design, generative art, video production, music composition, and technical visualization. Whether you're designing a landing page, creating educational content, or building interactive art, these skills provide professional-grade creative workflows.

## Available Skills

### Visual Design & Web

#### **claude-design**
Design one-off HTML artifacts like landing pages, pitch decks, and prototypes.

**Use when:** Creating polished web interfaces, marketing pages, or interactive prototypes without pre-existing design systems.

**Key features:**
- Complete design workflow from concept to implementation
- Responsive layouts with modern CSS
- Production-ready HTML artifacts
- Best practices for visual hierarchy and accessibility

---

#### **popular-web-designs**
54 real-world design systems (Stripe, Linear, Vercel, etc.) available as HTML/CSS templates.

**Use when:** You want your page to match the visual language of established brands.

**Key features:**
- Complete color palettes, typography, and component styles
- Exact CSS values from production sites
- Responsive behavior patterns
- Ready-to-use design tokens

---

#### **design-md**
Author, validate, and export Google's DESIGN.md token specification files.

**Use when:** Creating formal design system documentation or token specifications for handoff.

**Key features:**
- Structured design token format
- Validation and export tools
- Industry-standard DESIGN.md format
- Integration with design workflows

---

### Diagrams & Technical Visualization

#### **architecture-diagram**
Generate professional, dark-themed SVG architecture and infrastructure diagrams as standalone HTML.

**Use when:** Documenting system architecture, cloud infrastructure, or microservice topologies.

**Key features:**
- Dark grid-backed aesthetic
- Software architecture (frontend/backend/database)
- Cloud infrastructure (VPC, regions, services)
- No external dependencies (pure HTML/SVG)

---

#### **excalidraw**
Hand-drawn style diagrams (architecture, flowcharts, sequence) in Excalidraw JSON format.

**Use when:** You want informal, sketch-like diagrams that feel approachable and collaborative.

**Key features:**
- Hand-drawn aesthetic
- Architecture, flow, and sequence diagrams
- Excalidraw-compatible JSON output
- Collaborative editing support

---

### Infographics & Educational Content

#### **baoyu-infographic**
Create infographics with 21 layouts × 21 visual styles (信息图, 可视化).

**Use when:** Visualizing data, processes, or concepts for presentations and educational content.

**Key features:**
- 441 layout/style combinations
- Data visualization templates
- Educational and professional formats
- Chinese and English support

---

#### **baoyu-comic**
Knowledge comics (知识漫画) — educational, biographical, and tutorial content in comic format.

**Use when:** Explaining complex topics through visual storytelling or creating engaging educational materials.

**Key features:**
- Educational comic layouts
- Biography and tutorial formats
- Visual knowledge transfer
- Engaging narrative structure

---

### Generative Art & Interactive Media

#### **p5js**
Create p5.js sketches — generative art, shaders, interactive visuals, and 3D graphics.

**Use when:** Building interactive art, generative compositions, or creative coding projects.

**Key features:**
- Generative art algorithms
- WebGL shaders and 3D
- Interactive visualizations
- Creative coding workflows

---

#### **pixel-art**
Generate pixel art with authentic era palettes (NES, Game Boy, PICO-8).

**Use when:** Creating retro game assets, nostalgic visuals, or low-resolution art.

**Key features:**
- Era-accurate color palettes
- Pixel-perfect rendering
- Game Boy, NES, PICO-8 styles
- Sprite and tile generation

---

#### **ascii-art**
ASCII art generation using pyfiglet, cowsay, boxes, and image-to-ASCII conversion.

**Use when:** Creating text-based art, terminal banners, or converting images to ASCII.

**Key features:**
- Multiple ASCII art tools (pyfiglet, cowsay, boxes)
- Image-to-ASCII conversion
- Text banners and decorative elements
- Terminal-friendly output

---

### Video & Animation

#### **ascii-video**
Convert video/audio to colored ASCII animations as MP4 or GIF.

**Use when:** Creating retro-style video effects or terminal-based video playback.

**Key features:**
- Video-to-ASCII conversion
- Colored ASCII output
- MP4 and GIF export
- Audio synchronization

---

#### **manim-video**
Create 3Blue1Brown-style math and algorithm visualization videos using Manim CE.

**Use when:** Explaining mathematical concepts, algorithms, or creating educational animations.

**Key features:**
- Mathematical visualization engine
- Algorithm animation
- 3Blue1Brown aesthetic
- High-quality video export

---

### Music & Audio

#### **songwriting-and-ai-music**
Songwriting craft guidance and Suno AI music generation prompts.

**Use when:** Composing original music, generating AI music, or refining songwriting skills.

**Key features:**
- Songwriting technique guidance
- Suno AI prompt engineering
- Music composition workflows
- Creative music generation

---

### Creative Tools & Utilities

#### **humanizer**
Remove AI-isms and add authentic human voice to text.

**Use when:** Making AI-generated content sound more natural and relatable.

**Key features:**
- Strips generic AI patterns
- Adds personality and voice
- Maintains meaning while improving tone
- Natural language output

---

#### **creative-ideation** (ideation)
Generate project ideas through creative constraints and structured brainstorming.

**Use when:** Starting new creative projects or overcoming creative blocks.

**Key features:**
- Constraint-based idea generation
- Structured brainstorming frameworks
- Project concept development
- Creative prompts and exercises

---

#### **touchdesigner-mcp**
Control TouchDesigner instances via MCP — create operators, set parameters, wire connections, and build real-time visuals.

**Use when:** Building interactive installations, real-time video systems, or generative visuals with TouchDesigner.

**Key features:**
- 36 native TouchDesigner tools
- Operator creation and manipulation
- Parameter control and network building
- Real-time visual programming

---

## Quick Start

### Example: Design a Landing Page

```bash
# 1. Browse available design systems
/popular-web-designs

# 2. Design with specific brand style
/claude-design "Create a SaaS landing page in Stripe's style with hero, features, and pricing"
```

### Example: Create Educational Content

```bash
# 1. Generate knowledge comic
/baoyu-comic "Explain neural networks through a comic strip"

# 2. Or create an infographic
/baoyu-infographic "Timeline of AI breakthroughs with milestones"
```

### Example: Technical Documentation

```bash
# 1. Architecture diagram
/architecture-diagram "Microservices architecture with API gateway, auth service, and database"

# 2. Hand-drawn flow diagram
/excalidraw "User authentication flow from login to session creation"
```

### Example: Generative Art

```bash
# 1. Interactive p5.js sketch
/p5js "Generative landscape with Perlin noise and particle system"

# 2. Pixel art sprite
/pixel-art "8-bit character sprite in Game Boy palette"
```

## Skill Combinations

**Complete Design Workflow:**
1. Use `creative-ideation` to brainstorm concepts
2. Use `claude-design` or `popular-web-designs` for implementation
3. Use `design-md` to document the system

**Educational Content Pipeline:**
1. Use `baoyu-infographic` for data visualization
2. Use `baoyu-comic` for narrative explanation
3. Use `manim-video` for animated demonstrations

**Technical Documentation:**
1. Use `architecture-diagram` for system overview
2. Use `excalidraw` for collaborative workflows
3. Use `design-md` for design token specs

**Generative Art Project:**
1. Use `creative-ideation` for concept development
2. Use `p5js` or `pixel-art` for implementation
3. Use `ascii-video` or `manim-video` for animation

## Choosing the Right Skill

**For web design:**
- Quick prototype → `claude-design`
- Match existing brand → `popular-web-designs`
- Design system docs → `design-md`

**For diagrams:**
- Professional/dark theme → `architecture-diagram`
- Informal/collaborative → `excalidraw`
- ASCII/terminal → `ascii-art`

**For education:**
- Data visualization → `baoyu-infographic`
- Storytelling → `baoyu-comic`
- Math/algorithms → `manim-video`

**For art:**
- Interactive → `p5js`
- Retro/game → `pixel-art`
- ASCII → `ascii-art` or `ascii-video`

## Contributing

Found a bug or have an enhancement idea?

1. Open an issue with details
2. Fork the repository
3. Make changes to the relevant `SKILL.md`
4. Submit a pull request

## Related Categories

- **software-development/** - Coding and development workflows
- **productivity/** - General productivity tools
- **media/** - Media processing and manipulation
- **github/** - GitHub integration for creative projects

---

**Questions?** Check the [Hermes Agent documentation](https://hermes-agent.nousresearch.com/docs/) or ask in the [Discord community](https://discord.gg/nousresearch).
