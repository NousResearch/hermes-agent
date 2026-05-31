import { type CSSProperties } from 'react'
import { useStore } from '@nanostores/react'
import { Button } from '../components/button'
import {
  $logPath,
  openLogDir,
  startInstall,
  type BootstrapStateModel
} from '../store'
import { RefreshCw, FileText } from 'lucide-react'

interface FailureProps {
  bootstrap: BootstrapStateModel
}

/*
 * Failure screen. Same hero treatment as Welcome/Success — the wordmark
 * carries the brand, so we keep it across every terminal state.
 *
 * The actual error message lives below in muted text. Two clear
 * affordances: Retry (primary) and Open log folder (secondary).
 */
export default function Failure({ bootstrap }: FailureProps) {
  const logPath = useStore($logPath)

  return (
    <div className="hermes-fade-in flex h-full flex-col items-center justify-center gap-6 px-12 py-10">
      <div className="w-full max-w-2xl min-w-0 text-center">
        <p
          className="fit-text mx-auto mb-4 w-full font-['Collapse'] font-bold uppercase leading-[0.9] tracking-[0.08em] text-destructive mix-blend-plus-lighter dark:text-destructive/90"
          style={
            {
              '--fit-text-line-height': '0.9',
              '--fit-text-max': '5rem',
              '--fit-text-min': '2.25rem'
            } as CSSProperties
          }
        >
          <span>
            <span>Install didn&rsquo;t finish</span>
          </span>
          <span aria-hidden="true">Install didn&rsquo;t finish</span>
        </p>

        <p className="m-0 mx-auto max-w-xl text-center text-sm leading-normal tracking-tight text-muted-foreground">
          {bootstrap.error ?? 'Something went wrong during installation.'}
        </p>
      </div>

      <div className="flex items-center gap-3">
        <Button
          onClick={() => void startInstall()}
          size="lg"
          className="inline-flex items-center gap-2 px-6"
        >
          <RefreshCw size={16} />
          Retry install
        </Button>
        <Button
          variant="outline"
          size="lg"
          onClick={() => void openLogDir()}
          className="inline-flex items-center gap-2"
        >
          <FileText size={16} />
          Open log folder
        </Button>
      </div>

      {logPath && (
        <p className="max-w-lg text-center text-xs text-muted-foreground/70">
          Log: <code className="font-mono">{logPath}</code>
        </p>
      )}
    </div>
  )
}
