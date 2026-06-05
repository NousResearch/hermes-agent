# Audio Tail Continuity (Reels/Shorts)

Use this when the user reports abrupt audio cutoff at the end of a fixed-length export (e.g., 35s vertical reel).

## Symptom
- Video reaches full duration (e.g., 35.000s), but audible audio ends early.
- `ffprobe` shows audio stream shorter than container/video.

## Root Cause Pattern
- Complex `filter_complex` chain with VO/music ducking + `-t` can produce a shorter encoded audio stream than expected.
- Looping background music alone does not guarantee final muxed stream length.

## Reliable Fix Pattern
1. Build the mix as standalone PCM WAV at exact target length.
2. Verify WAV duration equals target (`ffprobe`).
3. Mux WAV as the only final audio stream in MP4.
4. Re-verify final MP4 has equal video/audio/container durations.

## Canonical Commands (adapt target duration as needed)

```bash
# 1) Pre-render exact-length mix
ffmpeg -y \
  -stream_loop -1 -i bg_music.m4a \
  -i voiceover.mp3 \
  -filter_complex "
    [0:a]atrim=0:35,asetpts=N/SR/TB,volume=0.58,afade=t=out:st=32.2:d=2.8[bg];
    [1:a]adelay=350|350,asetpts=N/SR/TB,volume=1.15,acompressor=threshold=-20dB:ratio=2.5:attack=15:release=220[vo];
    [vo]asplit=2[vo_sc][vo_mix];
    [bg][vo_sc]sidechaincompress=threshold=0.035:ratio=9:attack=12:release=260[bgduck];
    [bgduck][vo_mix]amix=inputs=2:duration=longest:weights='1 1.2':normalize=0,
    apad=pad_dur=35,atrim=0:35,afade=t=out:st=34.2:d=0.8,alimiter=limit=0.94[a]
  " \
  -map "[a]" -c:a pcm_s16le audio_mix.wav

# 2) Verify WAV duration
ffprobe -v error -show_entries format=duration -of default=nk=1:nw=1 audio_mix.wav

# 3) Mux into final MP4 (video-only source recommended)
ffmpeg -y \
  -i base_concat_video_only.mp4 \
  -i audio_mix.wav \
  -filter_complex "[0:v]scale=1080:1920,format=yuv420p[v]" \
  -map "[v]" -map 1:a -t 35 \
  -c:v libx264 -crf 18 -preset medium -r 30 -pix_fmt yuv420p \
  -c:a aac -b:a 224k -movflags +faststart final.mp4

# 4) Final duration QA
ffprobe -v error -show_entries format=duration -show_entries stream=index,codec_type,duration -of json final.mp4
```

## Delivery QA (Telegram)
- Always send final artifact immediately as:
  - `MEDIA:/absolute/path/to/final.mp4`
- Do this for every revision so user can review without extra prompt.
