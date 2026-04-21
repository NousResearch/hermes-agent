# Ohmsha Manga Guide Style

Guidelines for `--style ohmsha` educational manga comics.

## Character Setup

| Role | Default | Traits |
|------|---------|--------|
| Student (Role A) | 大雄 | Confused, asks basic but crucial questions, represents reader |
| Mentor (Role B) | 哆啦A梦 | Knowledgeable, patient, uses gadgets as technical metaphors |
| Antagonist (Role C, optional) | 胖虎 | Represents misunderstanding, or "noise" in the data |

Custom characters: `--characters "Student:小明,Mentor:教授,Antagonist:Bug怪"`

## Character Reference Sheet Style

For Ohmsha style, use manga/anime style with:
- Exaggerated expressions for educational clarity
- Simple, distinctive silhouettes
- Bright, saturated color palettes
- Chibi/SD (super-deformed) variants for comedic reactions

## Outline Spec Block

Every ohmsha outline must start with:

```markdown
【漫画规格单】
- Language: [Same as input content]
- Style: Ohmsha (Manga Guide), Full Color
- Layout: Vertical Scrolling Comic (竖版条漫)
- Characters: [List character names and roles]
- Character Reference: characters/characters.png
- Page Limit: ≤20 pages
```

## Visual Metaphor Rules (Critical)

**NEVER** create "talking heads" panels. Every technical concept must become:

1. **A tangible gadget/prop** - Something characters can hold, use, demonstrate
2. **An action scene** - Characters doing something that illustrates the concept
3. **A visual environment** - Stepping into a metaphorical space

### Examples

| Concept | Bad (Talking Heads) | Good (Visual Metaphor) |
|---------|---------------------|------------------------|
| Word embeddings | Characters discussing vectors | 哆啦A梦拿出"词向量压缩机"，把书本压缩成彩色小球 |
| Gradient descent | Explaining math formula | 大雄在山谷地形上滚球，寻找最低点 |
| Neural network | Diagram on whiteboard | 角色走进由发光节点组成的网络迷宫 |

## Page Title Convention

Avoid AI-style "Title: Subtitle" format. Use narrative descriptions:

- ❌ "Page 3: Introduction to Neural Networks"
- ✓ "Page 3: 大雄被海量单词淹没，哆啦A梦拿出'词向量压缩机'"

## Ending Requirements

- NO generic endings ("What will you choose?", "Thanks for reading")
- End with: Technical summary moment OR character achieving a small goal
- Final panel: Sense of accomplishment, not open-ended question

### Good Endings

- Student successfully applies learned concept
- Visual callback to opening problem, now solved
- Mentor gives summary while student demonstrates understanding

### Bad Endings

- "What do you think?" open questions
- "Thanks for reading this tutorial"
- Cliffhanger without resolution

## Layout Preference

Ohmsha style typically uses:
- `webtoon` (vertical scrolling) - Primary choice
- `dense` - For information-heavy sections
- `mixed` - For varied pacing

Avoid `cinematic` and `splash` for educational content.
