
#ifndef Py_INTERNAL_DICT_H
#define Py_INTERNAL_DICT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif


/* runtime lifecycle */

extern void _PyDict_Fini(PyInterpreterState *interp);


/* other API */

#ifndef WITH_FREELISTS
// without freelists
#  define PyDict_MAXFREELIST 0
#endif

#ifndef PyDict_MAXFREELIST
#  define PyDict_MAXFREELIST 80
#endif

struct _Py_dict_state {
#if PyDict_MAXFREELIST > 0
    /* Dictionary reuse scheme to save calls to malloc and free */
    PyDictObject *free_list[PyDict_MAXFREELIST];
    int numfree;
    PyDictKeysObject *keys_free_list[PyDict_MAXFREELIST];
    int keys_numfree;
#endif
};

typedef struct {
    /* Cached hash code of me_key. */
    Py_hash_t me_hash;
    PyObject *me_key;
    PyObject *me_value; /* This field is only meaningful for combined tables */
} PyDictKeyEntry;

typedef struct {
    PyObject *me_key;   /* The key must be Unicode and have hash. */
    PyObject *me_value; /* This field is only meaningful for combined tables */
} PyDictUnicodeEntry;

extern PyDictKeysObject *_PyDict_NewKeysForClass(void);
extern PyObject *_PyDict_FromKeys(PyObject *, PyObject *, PyObject *);

/* Gets a version number unique to the current state of the keys of dict, if possible.
 * Returns the version number, or zero if it was not possible to get a version number. */
extern uint32_t _PyDictKeys_GetVersionForCurrentState(PyDictKeysObject *dictkeys);

extern Py_ssize_t _PyDict_KeysSize(PyDictKeysObject *keys);

/* _Py_dict_lookup() returns index of entry which can be used like DK_ENTRIES(dk)[index].
 * -1 when no entry found, -3 when compare raises error.
 */
extern Py_ssize_t _Py_dict_lookup(PyDictObject *mp, PyObject *key, Py_hash_t hash, PyObject **value_addr);

extern Py_ssize_t _PyDict_GetItemHint(PyDictObject *, PyObject *, Py_ssize_t, PyObject **);
extern Py_ssize_t _PyDictKeys_StringLookup(PyDictKeysObject* dictkeys, PyObject *key);
extern PyObject *_PyDict_LoadGlobal(PyDictObject *, PyDictObject *, PyObject *);

/* Consumes references to key and value */
extern int _PyDict_SetItem_Take2(PyDictObject *op, PyObject *key, PyObject *value);
extern int _PyObjectDict_SetItem(PyTypeObject *tp, PyObject **dictptr, PyObject *name, PyObject *value);

extern PyObject *_PyDict_Pop_KnownHash(PyObject *, PyObject *, Py_hash_t, PyObject *);

#define DKIX_EMPTY (-1)
#define DKIX_DUMMY (-2)  /* Used internally */
#define DKIX_ERROR (-3)
#define DKIX_KEY_CHANGED (-4) /* Used internally */

typedef enum {
    DICT_KEYS_GENERAL = 0,
    DICT_KEYS_UNICODE = 1,
    DICT_KEYS_SPLIT = 2
} DictKeysKind;

/* See dictobject.c for actual layout of DictKeysObject */
struct _dictkeysobject {
    Py_ssize_t dk_refcnt;

    /* Size of the hash table (dk_indices). It must be a power of 2. */
    uint8_t dk_log2_size;

    /* Size of the hash table (dk_indices) by bytes. */
    uint8_t dk_log2_index_bytes;

    /* Kind of keys */
    uint8_t dk_kind;

    /* Version number -- Reset to 0 by any modification to keys */
    uint32_t dk_version;

    /* Number of usable entries in dk_entries. */
    Py_ssize_t dk_usable;

    /* Number of used entries in dk_entries. */
    Py_ssize_t dk_nentries;

    /* Actual hash table of dk_size entries. It holds indices in dk_entries,
       or DKIX_EMPTY(-1) or DKIX_DUMMY(-2).

       Indices must be: 0 <= indice < USABLE_FRACTION(dk_size).

       The size in bytes of an indice depends on dk_size:

       - 1 byte if dk_size <= 0xff (char*)
       - 2 bytes if dk_size <= 0xffff (int16_t*)
       - 4 bytes if dk_size <= 0xffffffff (int32_t*)
       - 8 bytes otherwise (int64_t*)

       Dynamically sized, SIZEOF_VOID_P is minimum. */
    char dk_indices[];  /* char is required to avoid strict aliasing. */

    /* "PyDictKeyEntry or PyDictUnicodeEntry dk_entries[USABLE_FRACTION(DK_SIZE(dk))];" array follows:
       see the DK_ENTRIES() macro */
};

/* This must be no more than 250, for the prefix size to fit in one byte. */
#define SHARED_KEYS_MAX_SIZE 30
#define NEXT_LOG2_SHARED_KEYS_MAX_SIZE 6

/* Layout of dict values:
 *
 * The PyObject *values are preceded by an array of bytes holding
 * the insertion order and size.
 * [-1] = prefix size. [-2] = used size. size[-2-n...] = insertion order.
 */
struct _dictvalues {
    PyObject *values[1];
};

#define DK_LOG_SIZE(dk)  ((dk)->dk_log2_size)
#if SIZEOF_VOID_P > 4
#define DK_SIZE(dk)      (((int64_t)1)<<DK_LOG_SIZE(dk))
#else
#define DK_SIZE(dk)      (1<<DK_LOG_SIZE(dk))
#endif
#define DK_ENTRIES(dk) \
    (assert(dk->dk_kind == DICT_KEYS_GENERAL), (PyDictKeyEntry*)(&((int8_t*)((dk)->dk_indices))[(size_t)1 << (dk)->dk_log2_index_bytes]))
#define DK_UNICODE_ENTRIES(dk) \
    (assert(dk->dk_kind != DICT_KEYS_GENERAL), (PyDictUnicodeEntry*)(&((int8_t*)((dk)->dk_indices))[(size_t)1 << (dk)->dk_log2_index_bytes]))
#define DK_IS_UNICODE(dk) ((dk)->dk_kind != DICT_KEYS_GENERAL)

extern uint64_t _pydict_global_version;

#define DICT_NEXT_VERSION() (++_pydict_global_version)

extern PyObject *_PyObject_MakeDictFromInstanceAttributes(PyObject *obj, PyDictValues *values);
extern PyObject *_PyDict_FromItems(
        PyObject *const *keys, Py_ssize_t keys_offset,
        PyObject *const *values, Py_ssize_t values_offset,
        Py_ssize_t length);

static inline void
_PyDictValues_AddToInsertionOrder(PyDictValues *values, Py_ssize_t ix)
{
    assert(ix < SHARED_KEYS_MAX_SIZE);
    uint8_t *size_ptr = ((uint8_t *)values)-2;
    int size = *size_ptr;
    assert(size+2 < ((uint8_t *)values)[-1]);
    size++;
    size_ptr[-size] = (uint8_t)ix;
    *size_ptr = size;
}

#ifdef __cplusplus
}
#endif
#endif   /* !Py_INTERNAL_DICT_H */
