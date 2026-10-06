# Jarvis — avatar de partículas

Avatar do Jarvis em WebGL: a imagem de referência é separada em camadas (fundo, corpo e
~220 mil partículas com a cor exata de cada pixel). Em repouso a cena reproduz a foto; mouse,
ondas, voz e estados (ouvindo / pensando / falando) deslocam partículas e corpo pela mesma
função na GPU, então tudo se move junto.

O busto é 3D (`src/model3d.js`), feito de sólidos de fatias elípticas medidas no desenho de
partículas da foto: a **cabeça** é rígida, e a mandíbula afina e avança, então o queixo fica na
frente do pescoço. O **pescoço e o tronco** formam outro sólido, atrás da cabeça, que torce e só
acompanha um pouco o giro. As **orelhas** são placas. Uma malha escura dos mesmos sólidos grava a
profundidade, e as partículas que ficam atrás dela (nuca, orelha do outro lado, pescoço atrás do
queixo) somem. O rosto tem relevo de manequim (`HEAD3D.FACE` e `HEAD3D.NOSE`): o nariz aparece de
perfil, as cavidades escurecem e as saliências clareiam. Da foto ficam o fundo e a aura.

## Rodar

Os módulos ES e as imagens soltas precisam de um servidor http (em `file://` o navegador bloqueia):

```bash
cd apps/jarvis-avatar
python3 -m http.server 8000      # abra http://localhost:8000
```

Microfone só funciona em `https://` ou `http://localhost`.

## Arquivo único (artifact / duplo clique)

```bash
python3 build.py                 # gera dist/jarvis.html (~5,4 MB, tudo embutido)
```

`dist/` não é versionado: é gerado a partir do código.

## Estrutura

| Caminho | O que tem |
|---|---|
| `index.html` | marcação da interface (status, dock, controles) |
| `src/styles.css` | visual + `@font-face` da Sora |
| `src/config.js` | `CFG` (ajustes), geometria da foto (`IMG`), lista de camadas, parâmetros de URL |
| `src/shaders.js` | GLSL: campo do mouse/ondas, pose, giro 3D (`place3D`), fundo, malha e partículas |
| `src/audio.js` | microfone, `<audio>` externo, voz do navegador → volume + 8 bandas |
| `src/pose.js` | balanço, olhadas, seguir o mouse, reação aos estados (molas amortecidas) |
| `src/model3d.js` | busto 3D: sólidos, partículas da superfície, relevo do rosto e malha de profundidade |
| `src/shapes.js` | formas: cabeça, esfera, galáxia, texto |
| `src/main.js` | WebGL, buffers, render (malha + partículas com profundidade), entrada, API pública, loop |
| `assets/img/` | camadas da foto (1672×941): `fundo`, `mascaras` e, pra aura em volta da figura, `emissao` e `info` |
| `assets/fonts/` | Sora 300/600/700 (SIL Open Font License 1.1) |
| `build.py` | empacota tudo em `dist/jarvis.html` (só biblioteca padrão) |

## Integração com o pipeline de voz

`window.JARVIS` existe depois que as camadas carregam; espere o evento `jarvis:ready`:

```js
addEventListener('jarvis:ready', ({ detail: jarvis }) => {
  jarvis.setState('thinking');           // 'idle' | 'listening' | 'thinking' | 'speaking' (null volta ao automático)
  jarvis.attachAudio(audioDoTTS);        // a boca e as partículas seguem o áudio real
});
```

| Chamada | Efeito |
|---|---|
| `setState(s)` | força um estado; qualquer outro valor devolve o controle ao automático |
| `attachAudio(el)` | analisa um `<audio>`/`<video>`; áudio de outra origem precisa de CORS, senão chega silêncio |
| `setLevel(0..1)` | volume manual por frame (quando o áudio não passa pelo navegador) |
| `speak(texto)` | fala com a voz do navegador (envelope de sílabas simulado) |
| `think(ms)` | estado "pensando" por `ms` (padrão 4500) |
| `shape(nome)` | `'head'`, `'sphere'`, `'galaxy'` ou `'text'` (usa o texto do campo) |
| `wave(x, y, força)` | onda a partir de um ponto, em px da imagem |
| `particles`, `time` | leitura: nº de partículas e segundos desde o início |

## Parâmetros de URL e atalhos

- `?estatico` — sem animação ambiente (comparar com a foto) · `?sofundo` — só o cenário, sem o humanoide
- `?fps` — mostra FPS · `?q=0.4..1` — resolução fixa (sem o ajuste automático para GPU fraca)
- `?giro=0.6` ou `?giro=0.6,0.1` — fixa o giro (e o aceno) em radianos: bancada para ajustar a cabeça 3D
- Teclas: **H** esconde a interface · **F** tela cheia · **M** microfone · **P** pensar · **espaço** fala ·
  **1–4** formas · **D** FPS
