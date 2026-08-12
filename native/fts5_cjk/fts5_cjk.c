/*
** fts5_cjk.c — "cjk_unicode61" FTS5 tokenizer: unicode61 + CJK bigrams.
**
** Why: SQLite's unicode61 tokenizer treats a CJK run as ONE token
** ("웅기가말했다" indexes as a single 6-char token), so a 2-char Korean
** query can never match inside it. The stock trigram tokenizer fixes
** substring search but needs >=3 chars per query term — 2-char Korean
** words (일본, 구글, 우리, ...) fall through to a full-table LIKE scan,
** measured at 3-6s per query on a 6.8GB messages table and the #1 driver
** of hermes session_search latency.
**
** What: wrap unicode61. Every token it emits is re-examined; maximal CJK
** runs inside the token are re-emitted as overlapping character BIGRAMS
** (Lucene CJKAnalyzer semantics), non-CJK segments pass through unchanged.
** A lone CJK char (run length 1) is emitted as a unigram. Because FTS5
** turns consecutive tokens emitted from one query term into a phrase,
** a query word like 캘린더 → [캘린][린더] gets exact substring semantics
** with index-speed lookups, down to 2-char terms.
**
** Build:  gcc -shared -fPIC -O2 fts5_cjk.c -o libfts5_cjk.so
** Load:   conn.load_extension(path)  # default entrypoint sqlite3_ftscjk_init
** Use:    CREATE VIRTUAL TABLE t USING fts5(c, tokenize='cjk_unicode61');
**         Extra args pass through to unicode61:
**         tokenize='cjk_unicode61 remove_diacritics 2'
*/
#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1

#include <string.h>
#include <stdlib.h>

/* ── CJK classification ──────────────────────────────────────────────── */

static int cjk_is_cjk(unsigned int cp) {
  return (cp >= 0xAC00 && cp <= 0xD7A3)      /* Hangul syllables        */
      || (cp >= 0x1100 && cp <= 0x11FF)      /* Hangul Jamo             */
      || (cp >= 0x3130 && cp <= 0x318F)      /* Hangul compat Jamo      */
      || (cp >= 0xA960 && cp <= 0xA97F)      /* Hangul Jamo ext-A       */
      || (cp >= 0xD7B0 && cp <= 0xD7FF)      /* Hangul Jamo ext-B       */
      || (cp >= 0x4E00 && cp <= 0x9FFF)      /* CJK unified ideographs  */
      || (cp >= 0x3400 && cp <= 0x4DBF)      /* CJK ext A               */
      || (cp >= 0xF900 && cp <= 0xFAFF)      /* CJK compat ideographs   */
      || (cp >= 0x20000 && cp <= 0x2FA1F)    /* CJK ext B..F, compat sup*/
      || (cp >= 0x3040 && cp <= 0x309F)      /* Hiragana                */
      || (cp >= 0x30A0 && cp <= 0x30FF)      /* Katakana                */
      || (cp >= 0x31F0 && cp <= 0x31FF);     /* Katakana phonetic ext   */
}

/* Decode one UTF-8 codepoint at p (n bytes available). Returns byte len
** consumed (>=1); stores codepoint in *pCp. Invalid bytes decode as
** themselves so segmentation still terminates. */
static int cjk_utf8_decode(const unsigned char *p, int n, unsigned int *pCp) {
  unsigned int c = p[0];
  if (c < 0x80) { *pCp = c; return 1; }
  if ((c & 0xE0) == 0xC0 && n >= 2) {
    *pCp = ((c & 0x1F) << 6) | (p[1] & 0x3F);
    return 2;
  }
  if ((c & 0xF0) == 0xE0 && n >= 3) {
    *pCp = ((c & 0x0F) << 12) | ((p[1] & 0x3F) << 6) | (p[2] & 0x3F);
    return 3;
  }
  if ((c & 0xF8) == 0xF0 && n >= 4) {
    *pCp = ((c & 0x07) << 18) | ((p[1] & 0x3F) << 12) |
           ((p[2] & 0x3F) << 6) | (p[3] & 0x3F);
    return 4;
  }
  *pCp = c;
  return 1;
}

/* ── tokenizer plumbing ──────────────────────────────────────────────── */

typedef struct CjkTokenizer CjkTokenizer;
struct CjkTokenizer {
  fts5_tokenizer inner;      /* unicode61 methods                  */
  Fts5Tokenizer *pInner;     /* unicode61 instance                 */
};

typedef struct CjkCallbackCtx CjkCallbackCtx;
struct CjkCallbackCtx {
  void *pOuterCtx;
  int (*xOuterToken)(void*, int, const char*, int, int, int);
};

/* Re-emit one unicode61 token, splitting CJK runs into bigrams.
**
** Offsets: unicode61 reports [iStart,iEnd) into the ORIGINAL text. For
** CJK bytes unicode61's folding is the identity, and ASCII case folding
** preserves byte length, so mapping sub-token offsets by byte position
** within the token is exact for CJK and correct-length for ASCII. For
** rare length-changing folds (accented latin) the highlight offsets can
** drift by a few bytes inside that token; matching is unaffected. Every
** emitted offset is clamped to [iStart,iEnd).
*/
static int cjk_emit(CjkCallbackCtx *p, int tflags,
                    const char *pToken, int nToken, int iStart, int iEnd) {
  const unsigned char *z = (const unsigned char*)pToken;
  int i = 0;
  int rc = SQLITE_OK;

  /* Fast path: no CJK anywhere → pass through untouched. */
  int hasCjk = 0;
  while (i < nToken) {
    unsigned int cp;
    i += cjk_utf8_decode(z + i, nToken - i, &cp);
    if (cjk_is_cjk(cp)) { hasCjk = 1; break; }
  }
  if (!hasCjk) {
    return p->xOuterToken(p->pOuterCtx, tflags, pToken, nToken, iStart, iEnd);
  }

#define CJK_CLAMP_END(v) ((iStart + (v)) > iEnd ? iEnd : (iStart + (v)))
  i = 0;
  while (i < nToken && rc == SQLITE_OK) {
    unsigned int cp;
    int segStart = i;
    int len = cjk_utf8_decode(z + i, nToken - i, &cp);
    if (!cjk_is_cjk(cp)) {
      /* non-CJK segment: extend to the next CJK char (or end) */
      i += len;
      while (i < nToken) {
        int l2 = cjk_utf8_decode(z + i, nToken - i, &cp);
        if (cjk_is_cjk(cp)) break;
        i += l2;
      }
      rc = p->xOuterToken(p->pOuterCtx, tflags,
                          pToken + segStart, i - segStart,
                          CJK_CLAMP_END(segStart), CJK_CLAMP_END(i));
    } else {
      /* CJK run: collect char byte-boundaries, emit bigrams. */
      int bounds[3];               /* rolling window: start, mid, end   */
      bounds[0] = segStart;
      bounds[1] = segStart + len;
      i += len;
      int nChars = 1;
      while (i < nToken) {
        int l2 = cjk_utf8_decode(z + i, nToken - i, &cp);
        if (!cjk_is_cjk(cp)) break;
        i += l2;
        nChars++;
        if (nChars >= 2) {
          bounds[2] = i;
          if (nChars == 2) {
            /* first bigram spans bounds[0]..bounds[2] */
          }
          rc = p->xOuterToken(p->pOuterCtx, tflags,
                              pToken + bounds[0], bounds[2] - bounds[0],
                              CJK_CLAMP_END(bounds[0]), CJK_CLAMP_END(bounds[2]));
          if (rc != SQLITE_OK) break;
          bounds[0] = bounds[1];
          bounds[1] = bounds[2];
        }
      }
      if (rc == SQLITE_OK && nChars == 1) {
        /* lone CJK char: emit as unigram */
        rc = p->xOuterToken(p->pOuterCtx, tflags,
                            pToken + segStart, bounds[1] - segStart,
                            CJK_CLAMP_END(segStart), CJK_CLAMP_END(bounds[1]));
      }
    }
  }
#undef CJK_CLAMP_END
  return rc;
}

static int cjkInnerCallback(void *pCtx, int tflags,
                            const char *pToken, int nToken,
                            int iStart, int iEnd) {
  return cjk_emit((CjkCallbackCtx*)pCtx, tflags, pToken, nToken, iStart, iEnd);
}

static int cjkCreate(void *pApiCtx, const char **azArg, int nArg,
                     Fts5Tokenizer **ppOut) {
  fts5_api *pApi = (fts5_api*)pApiCtx;
  CjkTokenizer *p;
  void *pInnerCtx = 0;
  int rc;

  p = (CjkTokenizer*)sqlite3_malloc(sizeof(CjkTokenizer));
  if (!p) return SQLITE_NOMEM;
  memset(p, 0, sizeof(*p));

  rc = pApi->xFindTokenizer(pApi, "unicode61", &pInnerCtx, &p->inner);
  if (rc == SQLITE_OK) {
    rc = p->inner.xCreate(pInnerCtx, azArg, nArg, &p->pInner);
  }
  if (rc != SQLITE_OK) {
    sqlite3_free(p);
    return rc;
  }
  *ppOut = (Fts5Tokenizer*)p;
  return SQLITE_OK;
}

static void cjkDelete(Fts5Tokenizer *pTok) {
  CjkTokenizer *p = (CjkTokenizer*)pTok;
  if (p) {
    if (p->pInner) p->inner.xDelete(p->pInner);
    sqlite3_free(p);
  }
}

static int cjkTokenize(Fts5Tokenizer *pTok, void *pCtx, int flags,
                       const char *pText, int nText,
                       int (*xToken)(void*, int, const char*, int, int, int)) {
  CjkTokenizer *p = (CjkTokenizer*)pTok;
  CjkCallbackCtx cb;
  cb.pOuterCtx = pCtx;
  cb.xOuterToken = xToken;
  return p->inner.xTokenize(p->pInner, &cb, flags, pText, nText,
                            cjkInnerCallback);
}

/* ── registration ────────────────────────────────────────────────────── */

static fts5_api *cjkFts5Api(sqlite3 *db) {
  fts5_api *pRet = 0;
  sqlite3_stmt *pStmt = 0;
  if (sqlite3_prepare_v2(db, "SELECT fts5(?1)", -1, &pStmt, 0) == SQLITE_OK) {
    sqlite3_bind_pointer(pStmt, 1, (void*)&pRet, "fts5_api_ptr", 0);
    sqlite3_step(pStmt);
  }
  sqlite3_finalize(pStmt);
  return pRet;
}

#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_ftscjk_init(sqlite3 *db, char **pzErrMsg,
                        const sqlite3_api_routines *pApi) {
  fts5_api *pFts;
  static fts5_tokenizer tok = { cjkCreate, cjkDelete, cjkTokenize };
  SQLITE_EXTENSION_INIT2(pApi);
  (void)pzErrMsg;
  pFts = cjkFts5Api(db);
  if (!pFts) {
    if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("fts5_cjk: FTS5 unavailable");
    return SQLITE_ERROR;
  }
  return pFts->xCreateTokenizer(pFts, "cjk_unicode61", (void*)pFts, &tok, 0);
}

/* Alias for callers that spell out the underscored basename. */
#ifdef _WIN32
__declspec(dllexport)
#endif
int sqlite3_fts5_cjk_init(sqlite3 *db, char **pzErrMsg,
                          const sqlite3_api_routines *pApi) {
  return sqlite3_ftscjk_init(db, pzErrMsg, pApi);
}
