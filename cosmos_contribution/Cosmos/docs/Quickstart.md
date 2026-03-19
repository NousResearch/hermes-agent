# cosmos Quick Start Guide

## One-Click Launch
Double-click `START.bat` for a menu with all options.

---

## Quick Commands

### Full CLI (All Features)
```bash
python main.py --cli
```
**Commands:** `help`, `memory`, `recall`, `swarm`, `evolve`, `dream`, `exit`

### Web Chat Interface
```bash
python run_web.py --demo
```
Opens at http://localhost:8080

### Emotional Visual Mode (CST + Camera + Mic)
```bash
python emotional_api/live_demo.py
```
**Two windows:** Live video feed + Data tokenization

### P2P Network Node
```bash
python main.py --node --dashboard
```

---

## CST Features Active

| Component | Description |
|-----------|-------------|
| **FrequencyAnalyzer** | Audio → Energy Mass (FFT) |
| **GeometricPhaseMapper** | Face → Phase Angle (68 pts) |
| **PhiInvariantEncoder** | Memory drift protection |
| **C_CONSTANT** | Swarm velocity limits |
| **cst_penalty()** | Evolution quality control |
| **EmotionalStateAPI** | Live mic/camera emotions |

---

## START.bat Menu

```
CORE MODES:
  1. Interactive CLI
  2. Web Chat
  3. Streamlit Dashboard

CST EMOTIONAL API:
  4. Emotional Visual Mode
  5. Visual Display Only
  6. CST Demo

NETWORK + ADVANCED:
  7. P2P Network Node
  8. P2P + Dashboard
  9. Health Dashboard

UTILITIES:
  10. Run All Tests
  11. First-Time Setup
```
