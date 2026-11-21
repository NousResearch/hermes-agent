import { useState } from 'react'
import { createRoot } from 'react-dom/client'

import { ActionsContextMenu } from '../components/ui/actions-menu'
import { PaneTab, PaneTabLabel, PaneTabStrip } from '../components/ui/pane-tab'

function Fixture() {
  const [closed, setClosed] = useState(false)
  const [activations, setActivations] = useState(0)

  const tab = (id: string, title: string) => (
    <PaneTab active data-testid={id} onClose={() => setClosed(true)}>
      <PaneTabLabel as="button" onClick={() => setActivations(value => value + 1)}>
        {title}
      </PaneTabLabel>
    </PaneTab>
  )

  return (
    <>
      <PaneTabStrip>
        {tab('bare', 'Meeting notes')}
        {!closed && (
          <ActionsContextMenu items={kit => <kit.Item>Rename</kit.Item>}>
            {tab('wrapped', 'Meeting notes')}
          </ActionsContextMenu>
        )}
        <ActionsContextMenu items={kit => <kit.Item>Rename</kit.Item>}>
          {tab('long', 'Plan a weekend visit to the city museum')}
        </ActionsContextMenu>
        <PaneTab data-testid="home">
          <PaneTabLabel>Home</PaneTabLabel>
        </PaneTab>
      </PaneTabStrip>
      <div style={{ height: 120, width: 28 }}>
        <PaneTab data-testid="vertical" onClose={() => undefined} vertical>
          <PaneTabLabel>Files</PaneTabLabel>
        </PaneTab>
      </div>
      <output data-testid="activations">{activations}</output>
    </>
  )
}

createRoot(document.getElementById('root')!).render(<Fixture />)
