export type IntroProps = {
  personality?: string
  seed?: number
}

const SPLASH_ALT =
  'Reuben Agent. Ask a question, paste an error, or point me at a repo. I can read code, run tools, and help you ship.'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

export function Intro(_props: IntroProps) {
  return (
    <div
      className="pointer-events-none flex w-full min-w-0 flex-col items-center justify-center px-4 py-6 text-center sm:px-8"
      data-slot="aui_intro"
    >
      <img
        alt={SPLASH_ALT}
        className="h-auto max-h-[min(70vh,42rem)] w-full max-w-[58rem] select-none object-contain dark:hidden"
        draggable={false}
        src={assetPath('reuben-splash.png')}
      />
      <img
        alt={SPLASH_ALT}
        className="hidden h-auto max-h-[min(70vh,42rem)] w-full max-w-[58rem] select-none object-contain dark:block"
        draggable={false}
        src={assetPath('reuben-splash-dark.png')}
      />
    </div>
  )
}
