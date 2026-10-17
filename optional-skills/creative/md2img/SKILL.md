---
name: md2img
description: Render Markdown as styled PNG images for chat.
version: 1.0.0
author: Juan Macias (@jmaciasluque), with Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [markdown, png, rendering, chat]
    category: creative
    related_skills: []
    config: {}
---

# md2img Skill

Use the external `md2img` CLI to render Markdown files as PNG images for chat
attachments and other image-only surfaces. This skill does not maintain the
renderer and does not support inline Markdown images.

## When to Use

- A chat client does not render Markdown tables or code blocks well.
- The user asks for a Markdown document, table, or snippet as an image.
- A styled PNG is easier to share than raw Markdown or HTML.

## Prerequisites

- Make sure the `md2img` executable is available on `PATH`.
- Use `terminal` to check the installed version:

  ```bash
  md2img -version
  ```

- If the command is missing, use `terminal` to install it with Homebrew on
  macOS or Linux:

  ```bash
  brew install jmaciasluque/tap/md2img
  ```

  Alternatively, install it with Go on any supported platform and make sure
  the Go binary directory is on `PATH`:

  ```bash
  go install github.com/jmaciasluque/md2img/cmd/md2img@latest
  ```

## How to Run

Use `terminal` and pass the Markdown input as a positional file argument:

```bash
md2img -o output.png input.md
```

For a tightly cropped chat attachment:

```bash
md2img -o output.png -trim -trim-padding 5 input.md
```

## Quick Reference

- `-o PATH`: Set the PNG output path.
- `-trim`: Crop unused whitespace around rendered content.
- `-trim-padding MM`: Set padding retained by `-trim`.
- `-dpi NUMBER`: Set output resolution; the default is 200.
- `-page-w MM`, `-page-h MM`: Set page dimensions.
- `-margin MM`: Set page margins.
- `-font NAME`, `-font-size POINTS`: Set body typography.
- `-heading-color HEX`, `-text-color HEX`: Set text colors.
- `-table-full-width`: Stretch tables to the available page width.

Input is positional, not `-input`. When no input path is supplied, the CLI
reads standard input; prefer a file for repeatable agent workflows.

## Procedure

1. Use `read_file` to inspect an existing Markdown source. If the content was
   supplied in chat, create a Markdown file with `patch`.
2. Choose an output path in the task workspace. Use `-trim` for chat-sized
   content and omit it when a fixed page is required.
3. Use `terminal` to run `md2img` with the input file and output path.
4. Inspect the generated PNG with `vision_analyze`. Check text size, wrapping,
   table width, code blocks, and clipping.
5. Adjust layout or typography flags and render again when the image is not
   legible. Return the final image using the platform's media attachment flow.

## Pitfalls

- Do not use shell pipelines or heredocs to construct the input. Use `patch`
  to create or update the Markdown file before rendering.
- Do not pass `-input`; the source file is the final positional argument.
- Inline Markdown images are not rendered. Keep inputs text-based.
- Font availability differs by machine. Use an installed font and inspect the
  result instead of assuming identical metrics across platforms.
- Wide tables and long unbroken strings can reduce readability. Increase page
  width, reduce font size, or simplify the source when needed.

## Verification

- Confirm the `terminal` call exits successfully and reports no render error.
- Open the PNG with `vision_analyze` and confirm all content is visible and
  readable at the intended sharing size.
- Confirm the output path is the final PNG that will be attached or returned.
