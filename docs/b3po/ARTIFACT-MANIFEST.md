# ARTIFACT-MANIFEST

| Campo | Windows (único artefato gerado nesta rodada) |
|---|---|
| Nome | `Hermes.exe` (nome de produto ainda **não** rebrandado) |
| Plataforma | Windows x64 |
| Versão do pacote | `0.0.0` (`apps/desktop/package.json`) |
| Commit | `21595add` + docs desta rodada (branch `b3po-brand`) |
| Caminho | `apps/desktop/release/win-unpacked/` |
| Formato | aplicativo desempacotado (`--dir`), **não** instalador |
| Tamanho | 235,4 MB |
| SHA-256 | `a570f6c626faf00a85738611d47c80792a722da1bde763b1a749fd4e12f62c43` |
| Assinatura | **NÃO ASSINADO** (`AZURE_SIGN_*` ausente — declarado pelo próprio empacotador) |
| Notarização | não se aplica (Windows) |
| Runner usado | Windows 11, Node v26.12.1, electron-builder |
| Limitações | não é instalador; nome ainda Hermes; sem assinatura; não testado em instalação limpa |
| Instalação | executar o `.exe` na pasta desempacotada (uso interno) |

## Artefatos ausentes (e por quê)

| Plataforma | Artefato | Motivo |
|---|---|---|
| Windows | MSIX/NSIS assinado | sem certificado e sem rodar `dist:win` nesta rodada |
| macOS | `.app`/`.dmg` | sem macOS/Xcode — **BLOCKED_BY_EXTERNAL_DEPENDENCY** |
| Linux | AppImage/deb/rpm | exige container Linux; viável, não executado |
| Android | APK/AAB | não existe projeto cliente mobile |
| iOS | archive/IPA | sem macOS/Xcode/certificado — **BLOCKED_BY_EXTERNAL_DEPENDENCY** |

Nenhum artefato acima pode ser anunciado como existente. O manifesto lista só o que foi produzido.
