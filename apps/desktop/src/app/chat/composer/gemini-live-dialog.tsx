import { useStore } from '@nanostores/react'
import * as React from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { triggerHaptic } from '@/lib/haptics'
import { Eye, EyeOff, Mic, Zap } from '@/lib/icons'
import {
  DEFAULT_GEMINI_LIVE_MODEL,
  DEFAULT_GEMINI_LIVE_VOICE,
  GEMINI_LIVE_MODELS,
  GEMINI_VOICES,
  getStoredGeminiLiveApiKey,
  getStoredGeminiLiveModel,
  getStoredGeminiLiveVoice,
  setStoredGeminiLiveApiKey,
  setStoredGeminiLiveModel,
  setStoredGeminiLiveVoice
} from '@/lib/gemini-live'
import { notify } from '@/store/notifications'
import {
  $geminiLiveDialogOpen,
  closeGeminiLiveDialog,
  setVoiceChatMode
} from '@/store/voice-live'

export function GeminiLiveDialog() {
  const open = useStore($geminiLiveDialogOpen)
  const [apiKey, setApiKey] = React.useState('')
  const [voice, setVoice] = React.useState(DEFAULT_GEMINI_LIVE_VOICE)
  const [model, setModel] = React.useState(DEFAULT_GEMINI_LIVE_MODEL)
  const [showKey, setShowKey] = React.useState(false)

  React.useEffect(() => {
    if (open) {
      setApiKey(getStoredGeminiLiveApiKey())
      setVoice(getStoredGeminiLiveVoice())
      setModel(getStoredGeminiLiveModel())
    }
  }, [open])

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault()
    triggerHaptic('submit')

    setStoredGeminiLiveApiKey(apiKey)
    setStoredGeminiLiveVoice(voice)
    setStoredGeminiLiveModel(model)
    void setVoiceChatMode('gemini-live')

    notify({
      id: 'gemini-live-configured',
      kind: 'success',
      message: 'Gemini Live voice configured successfully'
    })

    closeGeminiLiveDialog()
  }

  return (
    <Dialog onOpenChange={isOpen => (isOpen ? undefined : closeGeminiLiveDialog())} open={open}>
      <DialogContent bodyClassName="gap-5" className="max-w-md">
        <DialogHeader>
          <DialogTitle icon={Mic}>Gemini Live Voice Settings</DialogTitle>
          <DialogDescription>
            Full-duplex real-time voice powered directly by Google GenAI Multimodal Live
            WebSocket. Runs 100% locally on your laptop and delegates tasks to Hermes.
          </DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSave}>
          <div className="grid gap-1.5">
            <label className="text-xs font-medium text-foreground flex items-center justify-between">
              <span>Google / Gemini API Key</span>
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1 transition-colors"
              >
                {showKey ? <EyeOff className="size-3.5" /> : <Eye className="size-3.5" />}
                <span>{showKey ? 'Hide' : 'Show'}</span>
              </button>
            </label>
            <Input
              autoComplete="off"
              autoCorrect="off"
              onChange={e => setApiKey(e.target.value)}
              placeholder="AIzaSy..."
              spellCheck={false}
              type={showKey ? 'text' : 'password'}
              value={apiKey}
            />
            <p className="text-[11px] text-muted-foreground">
              Stored securely in your local Hermes Desktop browser storage. Works without needing
              changes on your remote VPS gateway.
            </p>
          </div>

          <div className="grid gap-1.5">
            <label className="text-xs font-medium text-foreground">Voice Persona</label>
            <Select onValueChange={setVoice} value={voice}>
              <SelectTrigger>
                <SelectValue placeholder="Select a voice persona" />
              </SelectTrigger>
              <SelectContent>
                {GEMINI_VOICES.map(v => (
                  <SelectItem key={v.id} value={v.id}>
                    <div className="flex flex-col text-left">
                      <span className="font-medium">{v.label}</span>
                      <span className="text-[11px] text-muted-foreground">{v.tone}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="grid gap-1.5">
            <label className="text-xs font-medium text-foreground">Model</label>
            <Select onValueChange={setModel} value={model}>
              <SelectTrigger>
                <SelectValue placeholder="Select a live model" />
              </SelectTrigger>
              <SelectContent>
                {GEMINI_LIVE_MODELS.map(m => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="rounded-md border border-border/50 bg-muted/40 p-2.5 text-xs text-muted-foreground flex items-start gap-2">
            <Zap className="size-4 shrink-0 text-primary mt-0.5" />
            <span>
              Gemini Live listens at 16kHz PCM, streams speech at 24kHz PCM, and supports instant
              barge-in cancellation. Any tool request is seamlessly delegated to Hermes.
            </span>
          </div>

          <DialogFooter>
            <Button onClick={closeGeminiLiveDialog} type="button" variant="ghost">
              Cancel
            </Button>
            <Button type="submit">Save & Activate</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
