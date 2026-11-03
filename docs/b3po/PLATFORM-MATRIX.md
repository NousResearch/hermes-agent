# PLATFORM-MATRIX — o que esta máquina consegue produzir

Fatos medidos em 2026-11-02, Windows 11 Pro (10.0.26300).

| Plataforma | Artefato esperado | Ferramentas presentes | Status real | Bloqueio |
|---|---|---|---|---|
| **Windows** | `.exe` / MSIX (NSIS conforme config) | Node 26, npm 12, electron-builder, JDK 21 | **VIÁVEL** | sem certificado de assinatura → SmartScreen |
| **macOS** | `.app`, `.dmg`/`.pkg` | nenhuma (sem Xcode, sem `sips`/`hdiutil`) | **IMPOSSÍVEL AQUI** | hardware Apple + Xcode + certificados |
| **iOS** | archive/IPA | nenhuma (sem Xcode) | **IMPOSSÍVEL AQUI** | macOS + Xcode + provisioning |
| **Linux** | AppImage, `.deb`, `.rpm` | Docker + WSL presentes | **VIÁVEL VIA CONTAINER** (não executado) | electron-builder não empacota Linux nativo no Windows |
| **Android** | APK/AAB | JDK 21, Android SDK em `tools\android\sdk`, `adb` | **SEM PROJETO** | não existe cliente mobile no repositório |

## Consequência para o status final

- Windows é a plataforma que pode chegar a **artefato + instalação comprovada** nesta máquina.
- macOS/iOS são **BLOCKED_BY_EXTERNAL_DEPENDENCY** por hardware/certificado — não é falha de código.
- Linux é viável, mas exige container; pode ser executado em rodada dedicada.
- Android/iOS dependem de decisão de arquitetura + cliente novo (ver `MOBILE-ARCHITECTURE-DECISION.md`).

## Como desbloquear

1. **macOS/iOS**: um Mac (físico ou runner macOS no GitHub Actions) + certificado Apple Developer.
2. **Android**: projeto cliente (Kotlin ou React Native) + Gradle wrapper; SDK já existe.
3. **Windows assinado**: certificado EV/OV Code Signing.
4. **Linux**: rodar `docker run` com a imagem do electron-builder e o script `dist:linux`.
