# Session note — One Piece Zoro sword anatomy + JoJo neon batch

## Context
Nick repeatedly regenerated One Piece / JoJo character posters in the glossy neon sticker-bomb style. Most batches succeeded, but Zoro required repeated correction because the model kept drifting on sword mechanics.

## Durable lesson: Zoro / three-sword users
For Roronoa Zoro or any character whose identity depends on multiple swords, treat weapon anatomy as a first-class constraint, not a background detail.

Failures observed:
- hand grips attached to the wrong part of the blade;
- handle / guard / blade junction melted into green slash effects;
- mouth sword appeared to grow from the face or lacked a clear handle/guard/blade sequence;
- extra blade shards appeared when the prompt emphasized circular sword motion too strongly;
- sticker-bomb text/effects crossed the hilt area and made the structure unreadable.

Working correction pattern:
1. Reduce pose complexity before retrying: stable half-body / upper-thigh, front three-quarter or side-profile stance beats extreme airborne twists.
2. Keep exactly three katanas and describe each separately:
   - mouth sword: handle at one mouth side, guard just outside cheek, straight blade extending horizontally outward;
   - left-hand sword: wrapped handle gripped by hand → visible round/oval guard → straight blade extending away from hand;
   - right-hand sword: same clean handle → guard → blade sequence.
3. Move sticker density to borders/background and keep hilt/guard/blade junctions unobscured.
4. Treat green slash trails as decorative energy behind the metal blades, never as replacements for the blades.
5. Add hard negatives: no hand gripping blade, no missing guard, no blade from wrong side of handle, no melted/fused handle, no extra handles, no random sword shards, no energy ribbon replacing blade, no stickers covering hilt/guard.

## One Piece popular/female batch notes
- For generic “4 popular One Piece characters,” a useful non-overlapping set after Straw Hats is Shanks / Trafalgar Law / Dracule Mihawk / Boa Hancock.
- For “4 popular female characters + Zoro,” a useful set is Nami / Nico Robin / Yamato / Perona + Zoro.
- Keep female characters covered/non-sexualized and push style density to background/props instead of exposed-skin emphasis.

## JoJo batch notes
A worked four-character JoJo set:
- Jotaro Kujo — low-angle fashion diagonal; cap + black uniform + chain + star/punch motifs.
- Dio Brando — side-profile split; blond hair + gold/yellow outfit + clock/time-stop motifs.
- Giorno Giovanna — circular motion ring; blond curled rolls + pink/purple suit + gold flower/life motifs.
- Josuke Higashikata — dutch-angle action; pompadour + dark uniform + diamond/repair motifs.

Keep the style generic: glossy neon cyber-pop, thick black comic outlines, sticker-bomb typography. Use IP names only for subject fidelity, not as third-party style dependencies.
