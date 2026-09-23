import type { CameraControlState, CameraIntent } from '../model'

export interface CameraControlsProps {
  dispatch(intent: CameraIntent): void
  focusedEntityLabel?: string
  state: CameraControlState
}

interface Control {
  label: string
  intent: CameraIntent
}

const CAMERA_CONTROLS: readonly Control[] = [
  { label: 'Rotate Left', intent: { kind: 'orbit', deltaAlpha: -0.22, deltaBeta: 0 } },
  { label: 'Rotate Right', intent: { kind: 'orbit', deltaAlpha: 0.22, deltaBeta: 0 } },
  { label: 'Tilt Up', intent: { kind: 'orbit', deltaAlpha: 0, deltaBeta: -0.12 } },
  { label: 'Tilt Down', intent: { kind: 'orbit', deltaAlpha: 0, deltaBeta: 0.12 } },
  { label: 'Pan North', intent: { kind: 'pan', deltaX: 0, deltaZ: -3 } },
  { label: 'Pan South', intent: { kind: 'pan', deltaX: 0, deltaZ: 3 } },
  { label: 'Pan East', intent: { kind: 'pan', deltaX: 3, deltaZ: 0 } },
  { label: 'Pan West', intent: { kind: 'pan', deltaX: -3, deltaZ: 0 } },
  { label: 'Zoom In', intent: { kind: 'zoom', delta: -6 } },
  { label: 'Zoom Out', intent: { kind: 'zoom', delta: 6 } }
]

function cameraStatus(state: CameraControlState, focusedEntityLabel?: string): string {
  const entity = focusedEntityLabel?.trim() || 'selected entity'

  if (state.following && state.focusedEntityKey) {
    return `Following ${entity}`
  }

  if (state.focusedEntityKey) {
    return `Focused on ${entity}`
  }

  return 'City overview'
}

export function CameraControls({ dispatch, focusedEntityLabel, state }: CameraControlsProps) {
  return (
    <section aria-label="Lunar City camera controls" className="lunar-city-camera-controls">
      <div className="grid grid-cols-2 gap-1">
        {CAMERA_CONTROLS.map(control => (
          <button key={control.label} onClick={() => dispatch(control.intent)} type="button">
            {control.label}
          </button>
        ))}
        <button
          disabled={!state.focusedEntityKey || !state.following}
          onClick={() => {
            if (state.focusedEntityKey) {
              dispatch({ kind: 'focus', entityKey: state.focusedEntityKey, follow: false })
            }
          }}
          type="button"
        >
          Stop Following
        </button>
        <button onClick={() => dispatch({ kind: 'return-to-city' })} type="button">
          Return to City
        </button>
      </div>
      <output aria-atomic="true" aria-label="Camera position" aria-live="polite" role="status">
        {cameraStatus(state, focusedEntityLabel)}
      </output>
    </section>
  )
}
