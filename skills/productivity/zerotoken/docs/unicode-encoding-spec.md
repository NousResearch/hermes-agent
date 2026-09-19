# Unicode 安全编码规范（项目执行版）

> 本文件是「Unicode 安全编码规范」在本仓库的执行细则。
> 生成或修改任何代码、配置、文档时，必须遵守本规范。
> 配套工具见 `scripts/` 目录；仓库历史问题与根因见 `SKILL.md` 的「F. Windows/PowerShell 环境适配 → 已知陷阱与解决方案」。

## 总目标

> 内部统一使用 Unicode，外部文本数据统一优先使用 UTF-8，
> 所有编码边界显式声明编码，不依赖默认字符集，不进行无意义的重复转码。

## 硬性规定（15 条）

1. 所有源码、配置文件、模板文件、JSON、CSV、日志和文本文件统一使用 UTF-8 编码。
2. 不要依赖操作系统、IDE、运行环境或系统区域设置的默认字符集。
3. 任何涉及字符串、文件、网络请求、HTTP、数据库、JSON 序列化/反序列化的地方，都必须明确使用 UTF-8。
4. 禁止在没有明确需求的情况下使用 GBK、GB2312、ANSI、Latin-1、Windows-1252 等编码。
5. 禁止出现 UTF-8 编码后再按 GBK、ANSI 或其他编码解码的情况。
6. 文件读取和写入时，应显式指定 UTF-8，而不是使用默认编码。
7. Web 页面统一声明 UTF-8，例如：`<meta charset="UTF-8">`
8. HTTP 接口涉及文本内容时，应正确声明 UTF-8，例如：`Content-Type: application/json; charset=utf-8`
9. 数据库应优先使用支持完整 Unicode 的字符集，例如 MySQL 使用 `utf8mb4`，同时确保数据库、表、字段和连接字符集保持一致。
10. JSON 中的中文应正常作为 Unicode 字符处理，不要为了「防止乱码」而进行不必要的重复转码。
11. 不要对已经是 Unicode 字符串的数据重复执行 encode/decode。
12. 如果代码中存在 Base64、URL Encoding、HTML Entity、Unicode Escape 等编码操作，要明确区分「字符编码」和「数据转义」，不要混用。
13. 修改已有项目时，先检查原有编码方式，避免因为强制转换造成已有数据损坏。
14. 如果无法确定外部输入的字符编码，不要猜测，应在代码中增加明确的编码检测、参数配置或异常处理。
15. 中文字符串、中文注释、中文文件名和中文接口数据都必须能够正确读取、存储、传输和显示。

## 编码链路检查

生成代码前，检查整个字符处理链路，确保每一个环节的编码一致：

```
输入数据 → 字符串处理 → 文件/数据库 → 网络传输 → API → 前端/终端显示
```

## 项目执行细则

### Python 文件读写

- 所有 `open()` 必须显式指定编码：读取用 `open(path, 'rb')` 二进制读后显式 decode，
  或 `open(path, 'r', encoding='utf-8')`；写入一律 `open(path, 'w', encoding='utf-8')`。
- **禁止** `errors='replace'` 静默替换损坏字符（会把中文无声变成 U+FFFD 替换字符）。
  解码失败时应按 UTF-8 → UTF-16 → GB18030 依次尝试，全部失败则抛错提示，
  见 `scripts/batch_edit.py` 的 `safe_read`。
- 读取历史遗留文件（可能为 UTF-16 或 GB18030）用 `scripts/safe_io.py` 的 `safe_read`，
  它会自动检测 BOM 并做安全解码；写入统一 UTF-8 无 BOM（`safe_write` / `safe_append`）。

### 控制台输出（Windows 中文环境）

- Python 3.7+：模块加载时显式 `sys.stdout.reconfigure(encoding='utf-8')`
  （`sys.stderr` 同理），不要依赖系统代码页（中文 Windows 默认 GBK/936）。
- 兜底：无法重配置的流用 `safe_print`（`scripts/safe_io.py`），保证永不抛
  `UnicodeEncodeError`。
- 终端显示：PowerShell 中配合 `chcp 65001` 查看中文输出；若仍乱码，属于终端显示层
  问题，文件本身编码正确，用 `read_file` 工具验证内容。
- **禁止**用 PowerShell `Add-Content` 向 UTF-8 文件追加中文（默认 GBK 写入会污染），
  改用 Python `open(path, 'a', encoding='utf-8')` 或 `safe_io.safe_append`。

### PowerShell 脚本（.ps1）

- **唯一例外**：`.ps1` 文件必须使用 **UTF-8 with BOM**。
  原因：Windows PowerShell 5.1（系统自带）对无 BOM 文件按 ANSI 代码页（GBK）解码，
  含中文的 UTF-8 无 BOM 脚本会乱码甚至解析异常。
  PowerShell 7+ 无此问题，但为兼容 5.1 统一带 BOM。
- 已由 `scripts/init_env.ps1` 示范（首字节 `EF BB BF`）。

### Node.js / HTTP

- 请求头/响应头显式声明编码，例如
  `'Content-Type': 'application/json; charset=utf-8'`。
- `fetch` / `readFileSync` / `writeFileSync` 显式传 `'utf-8'`；`JSON.stringify` 默认
  保留 Unicode（不要 `escape` 转义中文）。
- URL 编码、Base64、HTML Entity 属于**数据转义**，与字符编码无关，不得混用。

### JSON

- 中文作为普通 Unicode 字符处理：Python 用 `json.dumps(data, ensure_ascii=False)`，
  写入文件时 `encoding='utf-8'`；不要为了「防乱码」做重复转码。

### 数据库

- 如引入数据库，使用 `utf8mb4`，并确保数据库、表、字段、连接字符集一致。

## 仓库现有工具链

| 工具 | 用途 |
|------|------|
| `scripts/safe_io.py` | 安全读写（safe_read/safe_write/safe_append/write_result），safe_print 控制台兜底 |
| `scripts/fix_encoding.py` | 扫描/转换文件编码为 UTF-8（scan / preview / convert / check-replacement） |
| `scripts/detect_gbk_contamination.py` | 检测并修复 UTF-8 文件中的 GBK 污染（scan / inspect / fix） |
| `scripts/batch_edit.py` | 一次多编辑（原子替换），safe_read 不静默损坏 |
| `scripts/verify_output.py` | 验证结果写入 UTF-8 文件（替代 print），grep_check 显式解码 |
| `scripts/audit_encoding.py` | 全项目编码审计（UTF-8/BOM/替换字符/混合换行） |
| `scripts/init_env.ps1` | Windows 环境初始化（git quotepath、编码健康检查） |

## 生成代码后的安全检查

每次生成/修改代码后，额外执行一次：

```powershell
python scripts/audit_encoding.py --root . --out audit_result.txt
```

确认：
1. 无 `non-utf8` 文件；
2. 无 `替换字符`（U+FFFD）；
3. 无混合换行（LF/CRLF 混用）；
4. `.ps1` 文件带 BOM（审计单独列 `utf-8-sig` 属预期）。
