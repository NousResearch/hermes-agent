# Auto Selection

Content signals determine default art + tone + layout (or preset).

## Content Signal Matrix

| Content Signals | Art Style | Tone | Layout | Preset |
|-----------------|-----------|------|--------|--------|
| Tutorial, how-to, beginner | manga | neutral | webtoon | **ohmsha** |
| Computing, AI, programming | manga | neutral | dense | **ohmsha** |
| Technical explanation, educational | manga | neutral | webtoon | **ohmsha** |
| Pre-1950, classical, ancient | realistic | vintage | cinematic | - |
| Personal story, mentor | ligne-claire | warm | standard | - |
| Psychology, motivation, self-help, coaching | manga | warm | standard | **concept-story** |
| Business narrative, management, leadership | manga | warm | standard | **concept-story** |
| Conflict, breakthrough | (inherit) | dramatic | splash | - |
| Wine, food, lifestyle | realistic | neutral | cinematic | - |
| Martial arts, wuxia, xianxia | ink-brush | action | splash | **wuxia** |
| Romance, love, school life | manga | romantic | standard | **shoujo** |
| Business allegory, fable, parable, short insight, 四格 | minimalist | neutral | four-panel | **four-panel** |
| Biography, balanced | ligne-claire | neutral | mixed | - |

## Preset Recommendation Rules

**When preset is recommended**: Load `presets/{preset}.md` and apply all special rules.

### ohmsha
- **Triggers**: Tutorial, technical, educational, computing, programming, how-to, beginner
- **Special rules**: Visual metaphors, NO talking heads, gadget reveals, Doraemon-style characters
- **Base**: manga + neutral + webtoon/dense

### wuxia
- **Triggers**: Martial arts, wuxia, xianxia, cultivation, swordplay
- **Special rules**: Qi effects, combat visuals, atmospheric elements
- **Base**: ink-brush + action + splash

### shoujo
- **Triggers**: Romance, love story, school life, emotional drama
- **Special rules**: Decorative elements, eye details, romantic beats
- **Base**: manga + romantic + standard

### concept-story
- **Triggers**: Psychology, motivation, self-help, business narrative, management, leadership, personal growth, coaching, soft skills, abstract concept through story
- **Special rules**: Visual symbol system, growth arc, dialogue+action balance, original characters
- **Base**: manga + warm + standard

### four-panel
- **Triggers**: Business allegory, fable, parable, short insight, four-panel, 四格, 四格漫画, single-page comic, minimalist comic strip
- **Special rules**: Strict 起承转合 4-panel structure, B&W + spot color, simplified stick-figure characters, single-page story
- **Base**: minimalist + neutral + four-panel

## Compatibility Matrix

Art Style × Tone combinations work best when matched appropriately:

| Art Style | ✓✓ Best | ✓ Works | ✗ Avoid |
|-----------|---------|---------|---------|
| ligne-claire | neutral, warm | dramatic, vintage, energetic | romantic, action |
| manga | neutral, romantic, energetic, action | warm, dramatic | vintage |
| realistic | neutral, warm, dramatic, vintage | action | romantic, energetic |
| ink-brush | neutral, dramatic, action, vintage | warm | romantic, energetic |
| chalk | neutral, warm, energetic | vintage | dramatic, action, romantic |
| minimalist | neutral | warm, energetic | dramatic, vintage, romantic, action |

**Note**: Art Style × Tone × Layout can be freely combined. Incompatible combinations work but may produce unexpected results.

## Priority Order

1. User-specified options (`--art`, `--tone`, `--style`)
2. EXTEND.md defaults
3. Content signal analysis → auto-selection
4. Fallback: ligne-claire + neutral + standard
