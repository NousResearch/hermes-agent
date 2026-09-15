# Usage

This skill is triggered by natural language in Hermes — no slash command or CLI flags.

## Trigger Phrases

- "Illustrate this article" / "为文章配图"
- "Add images to this post"
- "Generate illustrations for [path/to/article.md]"

## Input Modes

| Mode | How to trigger | Output Directory |
|------|----------------|------------------|
| File path | Mention an article path (`path/to/article.md`) | `{article-dir}/imgs/` (default) |
| Pasted content | Paste the article text in the conversation | `illustrations/{topic-slug}/` (cwd) |

## Specifying Options in Natural Language

The user can specify any of the following directly in their request. If not specified, the skill asks via the `clarify` tool.

| Option | Example phrasing |
|--------|------------------|
| Type | "as an infographic", "as a flowchart", "as scenes" |
| Style | "in blueprint style", "use notion style", "用 watercolor 风格" |
| Preset | "use the tech-explainer preset", "storytelling preset" |
| Palette | "with macaron palette", "warm colors only" |
| Density | "minimal images", "one per section", "rich illustrations" |
| Language | "images in English" / "图片文字用中文" |
| Output | "save images alongside the article" / "put them in `illustrations/`" |

## Examples

**Technical article with data**:
> 帮我为 api-design.md 配图，用 infographic + blueprint 风格

**Preset shortcut**:
> Illustrate api-design.md with the tech-explainer preset

**Personal story**:
> Illustrate journey.md using the storytelling preset

**Tutorial with rich images**:
> Generate illustrations for how-to-deploy.md — tutorial preset, rich density

**Opinion article**:
> Illustrate opinion.md with the opinion-piece preset

**Preset with style override**:
> Use the tech-explainer preset for article.md but swap the style for notion
