#ifndef Py_INTERNAL_TYPEOBJECT_H
#define Py_INTERNAL_TYPEOBJECT_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif


/* runtime lifecycle */

extern PyStatus _PyTypes_InitState(PyInterpreterState *);
extern PyStatus _PyTypes_InitTypes(PyInterpreterState *);
extern void _PyTypes_FiniTypes(PyInterpreterState *);
extern void _PyTypes_Fini(PyInterpreterState *);


/* other API */

// Type attribute lookup cache: speed up attribute and method lookups,
// see _PyType_Lookup().
struct type_cache_entry {
    unsigned int version;  // initialized from type->tp_version_tag
    PyObject *name;        // reference to exactly a str or None
    PyObject *value;       // borrowed reference or NULL
};

#define MCACHE_SIZE_EXP 12
#define MCACHE_STATS 0

struct type_cache {
    struct type_cache_entry hashtable[1 << MCACHE_SIZE_EXP];
#if MCACHE_STATS
    size_t hits;
    size_t misses;
    size_t collisions;
#endif
};

extern PyStatus _PyTypes_InitSlotDefs(void);

extern void _PyStaticType_Dealloc(PyTypeObject *type);


#ifdef __cplusplus
}
#endif
#endif /* !Py_INTERNAL_TYPEOBJECT_H */
