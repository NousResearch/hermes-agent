# Audio

## Importing Audio

```tsx
import { Audio, staticFile } from "remotion";

// Background music (full duration)
<Audio src={staticFile("music.mp3")} volume={0.3} />

// Voiceover (starts at frame 60)
<Sequence from={60}>
  <Audio src={staticFile("voiceover.mp3")} volume={1.0} />
</Sequence>
```

Audio files MUST be in `public/` and referenced via `staticFile()`.

### Volume Control

```tsx
// Static volume
<Audio src={staticFile("music.mp3")} volume={0.4} />

// Dynamic volume (duck music during voiceover)
<Audio
  src={staticFile("music.mp3")}
  volume={(f) =>
    interpolate(f, [voiceStart - 15, voiceStart, voiceEnd, voiceEnd + 15],
      [0.4, 0.1, 0.1, 0.4], { extrapolateLeft: "clamp", extrapolateRight: "clamp" })
  }
/>
```

### Trimming Audio

```tsx
<Audio src={staticFile("song.mp3")} startFrom={90} endAt={900} />
// Plays from frame 90 to frame 900 of the audio file (3s to 30s at 30fps)
```

## Audio Duration Detection

```tsx
import { getAudioDurationInSeconds } from "@remotion/media-utils";
import { staticFile } from "remotion";

// In calculateMetadata:
const duration = await getAudioDurationInSeconds(staticFile("voiceover.mp3"));
const durationInFrames = Math.ceil(duration * fps) + 90; // +3s padding
```

## Audio Visualization

```bash
npx remotion add @remotion/media-utils
```

### Windowed Audio Data

Get frequency/amplitude data for the current frame:

```tsx
import { useWindowedAudioData } from "@remotion/media-utils";
import { staticFile, useCurrentFrame, useVideoConfig } from "remotion";

const frame = useCurrentFrame();
const { fps } = useVideoConfig();

const { audioData, dataOffsetInSeconds } = useWindowedAudioData({
  src: staticFile("music.wav"),
  frame,
  fps,
  numberOfSamples: 256,     // FFT resolution
  windowInSeconds: 1 / fps, // one frame's worth of audio
});
```

`audioData` contains frequency amplitude values (0-255) for each FFT bin.

### Spectrum Bars

```tsx
const SpectrumBars: React.FC<{ audioSrc: string; barCount?: number }> = ({
  audioSrc,
  barCount = 32,
}) => {
  const frame = useCurrentFrame();
  const { fps, height } = useVideoConfig();

  const { audioData } = useWindowedAudioData({
    src: staticFile(audioSrc),
    frame,
    fps,
    numberOfSamples: 256,
    windowInSeconds: 0.1,
  });

  if (!audioData) return null;

  const barWidth = 100 / barCount;

  return (
    <AbsoluteFill style={{ alignItems: "flex-end" }}>
      <div style={{ display: "flex", width: "80%", height: "60%", alignItems: "flex-end", gap: 4 }}>
        {Array.from({ length: barCount }).map((_, i) => {
          const binIndex = Math.floor((i / barCount) * (audioData.length / 2));
          const amplitude = (audioData[binIndex] ?? 0) / 255;

          return (
            <div
              key={i}
              style={{
                flex: 1,
                height: `${amplitude * 100}%`,
                backgroundColor: `hsl(${(i / barCount) * 200 + 200}, 80%, 60%)`,
                borderRadius: 4,
                transition: "none", // NEVER use CSS transition
              }}
            />
          );
        })}
      </div>
    </AbsoluteFill>
  );
};
```

### Bass-Reactive Scale

Drive element scale by bass amplitude:

```tsx
const bassAmplitude = audioData
  ? audioData.slice(0, 8).reduce((sum, v) => sum + v, 0) / (8 * 255)
  : 0;

const scale = 1 + bassAmplitude * 0.3; // 1.0 to 1.3 range
<div style={{ transform: `scale(${scale})` }}>
  <Logo />
</div>
```

## Voiceover Generation

### ElevenLabs (Primary — highest quality)

Generate voiceover audio via the ElevenLabs API:

```typescript
// generate-voiceover.ts (run with: npx tsx generate-voiceover.ts)
import fs from "fs";

const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const VOICE_ID = "21m00Tcm4TlvDq8ikWAM"; // "Rachel" — default female

interface SceneScript {
  id: string;
  text: string;
}

const scenes: SceneScript[] = [
  { id: "intro", text: "Introducing our latest feature..." },
  { id: "feature", text: "Built from the ground up for performance." },
  { id: "cta", text: "Try it today. Link in the description." },
];

async function generateVoiceover(scene: SceneScript) {
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}`,
    {
      method: "POST",
      headers: {
        "xi-api-key": ELEVENLABS_API_KEY!,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: scene.text,
        model_id: "eleven_turbo_v2_5",
        voice_settings: { stability: 0.5, similarity_boost: 0.75 },
      }),
    }
  );

  const buffer = await response.arrayBuffer();
  fs.writeFileSync(`public/audio/${scene.id}.mp3`, Buffer.from(buffer));
  console.log(`Generated: public/audio/${scene.id}.mp3`);
}

for (const scene of scenes) {
  await generateVoiceover(scene);
}
```

### Qwen3-TTS (Free / Local Alternative)

For users without an ElevenLabs key, Qwen3-TTS runs locally:

```bash
# Install
pip install transformers torch torchaudio

# Generate via Python
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchaudio, torch

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-TTS', torch_dtype=torch.float16, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-TTS')

text = 'Introducing our latest feature.'
inputs = tokenizer(text, return_tensors='pt').to(model.device)
audio = model.generate(**inputs, max_new_tokens=2048)
torchaudio.save('public/audio/intro.wav', audio.cpu(), 24000)
"
```

Quality is lower than ElevenLabs but serviceable and free.

## Music Sync Patterns

### Beat-Aligned Cuts

If you know the BPM of the music track, calculate beat positions:

```tsx
const bpm = 120;
const framesPerBeat = (60 / bpm) * fps; // 15 frames per beat at 120bpm/30fps

// Scene durations as multiples of the beat
const scene1Duration = framesPerBeat * 8;  // 2 bars
const scene2Duration = framesPerBeat * 16; // 4 bars
```

### Manual Beat Markers

For irregular rhythms, define beat timestamps manually:

```tsx
const beatFrames = [0, 15, 30, 52, 67, 90, 105, 120]; // frames where beats land

// Flash or pulse on beats
const isNearBeat = beatFrames.some((b) => Math.abs(frame - b) < 3);
const beatPulse = isNearBeat ? 1.1 : 1.0;
```
