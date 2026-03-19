# Emotional State API - Quick Start

## One-Click Start
Double-click `START.bat` for a menu.

## Command Line

### Demo Mode
```bash
python emotional_state_api.py
```

### With Files
```bash
python emotional_state_api.py audio.wav face.jpg
python emotional_state_api.py none face.jpg    # Image only
python emotional_state_api.py audio.wav none   # Audio only
```

## Python Usage

```python
from emotional_state_api import EmotionalStateAPI

api = EmotionalStateAPI()
result = api.get_state("audio.wav", "face.jpg")

print(result["derived_emotion"])  # HAPPY | SAD | ANGRY | NEUTRAL
print(result["physics_data"])     # {"frequency_mass": 0.8, "geometric_phase": 0.3}
```

## CST Physics

| State | Mass | Phase | Description |
|-------|------|-------|-------------|
| HAPPY | > 0.7 | < 0.5 | High Energy + Low Tension |
| ANGRY | > 0.7 | > 0.5 | High Energy + High Tension |
| SAD | < 0.4 | any | Low Energy |
| NEUTRAL | else | else | Waiting for signal |
