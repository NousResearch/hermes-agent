# Reel remix audio continuity (VO + reference music)

Use this when rebuilding a new visual cut while preserving narration quality and musical continuity from a reference reel.

## Goal
- Keep **VO continuous and intelligible** (e.g., ElevenLabs track).
- Use **one continuous background music bed** from the reference reel.
- Do **not** mix per-clip source audios from generated/stock clips.

## Recommended workflow

1. **Prepare video-only base** (strip clip audio):
```bash
ffmpeg -y -i base_concat.mp4 -map 0:v:0 -c:v copy base_video_only.mp4
```

2. **Extract music from reference reel**:
```bash
ffmpeg -y -i reference_reel.mp4 -vn -ac 2 -ar 48000 \
  -af "loudnorm=I=-19:TP=-2:LRA=10" ref_music.m4a
```
- If reference is shorter than final video, loop with `-stream_loop -1` and `atrim` to final duration.

3. **Mix music + VO with ducking** (music dips under speech):
```bash
ffmpeg -y -i base_video_only.mp4 -stream_loop -1 -i ref_music.m4a -i voiceover.mp3 \
  -filter_complex "
    [1:a]atrim=0:35,afade=t=out:st=33:d=2,volume=0.58[bg];
    [2:a]adelay=350|350,volume=1.15,acompressor=threshold=-20dB:ratio=2.5:attack=15:release=220[vo_pre];
    [vo_pre]asplit=2[vo_sc][vo_mix];
    [bg][vo_sc]sidechaincompress=threshold=0.035:ratio=9:attack=12:release=260[bgduck];
    [bgduck][vo_mix]amix=inputs=2:duration=first:weights='1 1.2':normalize=0,alimiter=limit=0.94[a]
  " \
  -map 0:v -map "[a]" -t 35 -c:v libx264 -crf 18 -preset medium -r 30 -pix_fmt yuv420p \
  -c:a aac -b:a 224k -movflags +faststart final.mp4
```

## QA checks before delivery
- Confirm **no clip audio leakage** (base must be video-only).
- Confirm VO track is present and audible throughout speech window.
- Confirm music is perceptible during non-speech gaps and remains continuous.

## Common pitfalls
- `-an` in final export command silently removes all audio.
- Reusing `base_concat.mp4` with embedded clip audio causes muddy/uncontrolled background.
- Using only `amix` without ducking often makes either VO too buried or music too low.
