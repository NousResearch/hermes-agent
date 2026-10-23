import './styles.css'

import { installBrowserHost } from './bridge'
import { initializeBrowserMessages } from './messages'

installBrowserHost()
void initializeBrowserMessages().then(() => import('../src/main'))
