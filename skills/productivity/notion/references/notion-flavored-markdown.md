# Notion-Flavored Markdown

This reference covers the Notion-specific extensions supported by the `/markdown` endpoints.

Standard CommonMark works first. Load this file only when you need Notion-specific constructs.

## Formatting model

- Standard Markdown is supported.
- Notion adds XML-like tags for some block types.
- Use **tabs** for indentation in nested Notion-flavored constructs.

## Block-level constructs

### Callout

```md
<callout icon="🎯" color="blue_bg">
	Ship the MVP by **Friday**.
</callout>
```

### Toggle / details

```md
<details color="gray">
<summary>Toggle title</summary>
	Children indented one tab
</details>
```

### Columns

```md
<columns>
	<column>Left side</column>
	<column>Right side</column>
</columns>
```

### Table of contents

```md
<table_of_contents color="gray"/>
```

## Inline constructs

### Mentions

- User: `<mention-user url="..."/>`
- Page: `<mention-page url="...">Title</mention-page>`
- Date: `<mention-date start="2026-05-15"/>`

### Underline

```md
<span underline="true">text</span>
```

### Color

```md
<span color="blue">text</span>
```

Or block-level color on the first line when supported.

### Math

- Inline: `$x^2$`
- Block:

```md
$$
a^2 + b^2 = c^2
$$
```

### Citations

```md
[^https://example.com]
```

## Colors

Supported color names:
- `gray`
- `brown`
- `orange`
- `yellow`
- `green`
- `blue`
- `purple`
- `pink`
- `red`
- plus `*_bg` background variants

## Small gotchas

- Headings 5 and 6 collapse to H4.
- Multiple `>` lines render as separate quote blocks; use `<br>` inside one quote for a multi-line quote.
- If plain Markdown is enough, prefer plain Markdown. It is easier to generate and maintain.
