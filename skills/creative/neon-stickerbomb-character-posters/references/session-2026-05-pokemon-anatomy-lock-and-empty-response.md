# Pokemon neon sticker-bomb generation: anatomy lock + empty-response handling

Session: 2026-05-30
Skill: `neon-stickerbomb-character-posters`

## What happened

Nick generated and rerolled multiple rare Pokemon in the neon sticker-bomb style: Pikachu, Blastoise, Charmander, Mewtwo, Mew, Rayquaza, Lugia, Articuno, Moltres, Suicune, Raikou, Zapdos.

Two durable lessons emerged:

1. **Pokemon/creature anatomy must be treated as a hard fidelity gate.**
   - Rayquaza initially looked strong stylistically but lacked its paired small forearms/clawed hands.
   - The prompt said `small arm fins/claws if visible`, which was too weak and allowed the model to omit the hands.
   - Corrected wording required both small forearms/clawed hands visible behind the head, in a clean area, not replaced by fins/spikes/lightning/stickers.
   - Vision QC confirmed the corrected Rayquaza had visible paired foreclaws, continuous body, no duplicated head, and no broken body.

2. **Some exact Pokemon prompts can repeatedly return `empty_response`.**
   - Mewtwo repeatedly returned `empty_response` across long, compact, and ultra-compact prompts, including genericized `psychic clone alien` wording.
   - Zapdos initially returned `empty_response` twice, then later succeeded with an anatomy-locked direct prompt.
   - When a named slot repeatedly fails, preserve successful sibling outputs and write a partial manifest. Do not reuse older paths as fresh rerolls.

## Prompting corrections

### Rayquaza / 烈空坐

Use hard anatomy wording, not optional wording:

```text
ANATOMY LOCK: one long continuous serpentine emerald body, one dragon head with horn/fin cues, yellow ring markings, red mouth accents, one continuous tail, and a clearly visible matching pair of small forearms / clawed hands located just behind the head on both sides of the upper body. BOTH SMALL HANDS MUST BE VISIBLE: left and right forearms, each with small clawed fingers; do not replace them with fins, spikes, lightning, ribbons, or stickers. Body segments continuous, no duplicated heads, no broken body, no disconnected tail, no extra unrelated limbs.
```

Composition guidance:

```text
Rayquaza coils through the frame in a large readable spiral, head large in the upper-left third, both small clawed hands displayed below/behind the head in a clean open area, body looping around the frame. Typography follows the spiral path but does not cover the head or hands.
```

Negative block additions:

```text
no duplicated heads, no broken/fused body, no missing tail, no missing hands, no hidden hands, no hands replaced by fins, no hands covered by text/stickers, no extra limbs, no malformed claws, no stickers covering face or forearms
```

### Mewtwo / 超梦

Repeated failures suggest using progressive prompt compaction and, if necessary, generic visual identity wording. Still keep anatomy hard:

```text
ANATOMY LOCK: exactly two arms, exactly two legs, matching left/right three-finger hands, one huge thick purple tail clearly attached behind hips, purple abdomen plate, smooth feline alien head, narrow intense eyes, readable wrists elbows knees shoulders and tail base; no extra/missing/fused limbs, no five-finger hands, no mismatched hands.
```

If direct subject naming fails, compact to:

```text
Portrait neon stickerbomb poster: pale gray psychic clone alien beast, purple abdomen, one huge purple tail, exactly two arms/two legs, matching three-finger hands, smooth feline head, glowing orb, side-profile levitation, glossy black outlines, cyan magenta lights, acid green glow, graffiti tape barcode halftone collage, PSYCHIC CLONE text, NickZag tape tag. No five fingers, no mismatched hands, no extra limbs.
```

Do not call an older Mewtwo path a new generation. If all attempts fail, manifest the failed slot and optionally provide the prior path labeled as a reference only.

### Zapdos / 闪电鸟

Direct anatomy-locked wording eventually succeeded:

```text
ANATOMY LOCK: exactly one angular bird head, one beak, exactly two matching spiky wings, two talon legs, correct spiky tail feathers, readable shoulders and wing joints; no extra wings, no missing wings, no fused wings, no wrong limb count.
```

Keep the head and both wings unobscured; push sticker density to borders/background/hazard labels.

## Manifest / delivery rule

For partial rerolls like `重新生成闪电鸟 超梦 烈空坐`:

- Save a manifest with the requested order and per-slot status.
- Mark failed slots as `failed_empty_response_after_N_attempts`.
- Include `prior_successful_reference_path` only as a reference, never as the new result.
- Deliver only newly successful `MEDIA:` attachments.

## QC rule

For creature Pokemon, style success is not sufficient. Before reporting success, check:

- Does the named creature remain recognizable without text labels?
- Are required paired appendages visible and symmetric enough for the species?
- Are hands/foreclaws/paws/wings/tails present when canon requires them?
- Are appendages hidden by typography/stickers? If yes, reroll with a clean anatomy area.
