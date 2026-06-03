import Widget from './components/Widget';
import api from '@/lib/api';
import { formatName } from './utils';
import Screen from './screens';

export default function App() {
  return Widget({ api, formatName, Screen });
}
