#ifndef Py_INTERNAL_HASHTABLE_H
#define Py_INTERNAL_HASHTABLE_H
#ifdef __cplusplus
extern "C" {
#endif

#ifndef Py_BUILD_CORE
#  error "this header requires Py_BUILD_CORE define"
#endif

/* Single linked list */

typedef struct _Py_slist_item_s {
    struct _Py_slist_item_s *next;
} _Py_slist_item_t;

typedef struct {
    _Py_slist_item_t *head;
} _Py_slist_t;

#define _Py_SLIST_ITEM_NEXT(ITEM) (((_Py_slist_item_t *)ITEM)->next)

#define _Py_SLIST_HEAD(SLIST) (((_Py_slist_t *)SLIST)->head)


/* _Py_hashtable: table entry */

typedef struct {
    /* used by _Py_hashtable_t.buckets to link entries */
    _Py_slist_item_t _Py_slist_item;

    Py_uhash_t key_hash;
    void *key;
    void *value;
} _Py_hashtable_entry_t;


/* _Py_hashtable: prototypes */

/* Forward declaration */
struct _Py_hashtable_t;
typedef struct _Py_hashtable_t _Py_hashtable_t;

typedef Py_uhash_t (*_Py_hashtable_hash_func) (const void *key);
typedef int (*_Py_hashtable_compare_func) (const void *key1, const void *key2);
typedef void (*_Py_hashtable_destroy_func) (void *key);
typedef _Py_hashtable_entry_t* (*_Py_hashtable_get_entry_func)(_Py_hashtable_t *ht,
                                                               const void *key);

typedef struct {
    // Allocate a memory block
    void* (*malloc) (size_t size);

    // Release a memory block
    void (*free) (void *ptr);
} _Py_hashtable_allocator_t;


/* _Py_hashtable: table */
struct _Py_hashtable_t {
    size_t nentries; // Total number of entries in the table
    size_t nbuckets;
    _Py_slist_t *buckets;

    _Py_hashtable_get_entry_func get_entry_func;
    _Py_hashtable_hash_func hash_func;
    _Py_hashtable_compare_func compare_func;
    _Py_hashtable_destroy_func key_destroy_func;
    _Py_hashtable_destroy_func value_destroy_func;
    _Py_hashtable_allocator_t alloc;
};

/* Hash a pointer (void*) */
PyAPI_FUNC(Py_uhash_t) _Py_hashtable_hash_ptr(const void *key);

/* Comparison using memcmp() */
PyAPI_FUNC(int) _Py_hashtable_compare_direct(
    const void *key1,
    const void *key2);

PyAPI_FUNC(_Py_hashtable_t *) _Py_hashtable_new(
    _Py_hashtable_hash_func hash_func,
    _Py_hashtable_compare_func compare_func);

PyAPI_FUNC(_Py_hashtable_t *) _Py_hashtable_new_full(
    _Py_hashtable_hash_func hash_func,
    _Py_hashtable_compare_func compare_func,
    _Py_hashtable_destroy_func key_destroy_func,
    _Py_hashtable_destroy_func value_destroy_func,
    _Py_hashtable_allocator_t *allocator);

PyAPI_FUNC(void) _Py_hashtable_destroy(_Py_hashtable_t *ht);

PyAPI_FUNC(void) _Py_hashtable_clear(_Py_hashtable_t *ht);

typedef int (*_Py_hashtable_foreach_func) (_Py_hashtable_t *ht,
                                           const void *key, const void *value,
                                           void *user_data);

/* Call func() on each entry of the hashtable.
   Iteration stops if func() result is non-zero, in this case it's the result
   of the call. Otherwise, the function returns 0. */
PyAPI_FUNC(int) _Py_hashtable_foreach(
    _Py_hashtable_t *ht,
    _Py_hashtable_foreach_func func,
    void *user_data);

PyAPI_FUNC(size_t) _Py_hashtable_size(const _Py_hashtable_t *ht);

/* Add a new entry to the hash. The key must not be present in the hash table.
   Return 0 on success, -1 on memory error. */
PyAPI_FUNC(int) _Py_hashtable_set(
    _Py_hashtable_t *ht,
    const void *key,
    void *value);


/* Get an entry.
   Return NULL if the key does not exist. */
static inline _Py_hashtable_entry_t *
_Py_hashtable_get_entry(_Py_hashtable_t *ht, const void *key)
{
    return ht->get_entry_func(ht, key);
}


/* Get value from an entry.
   Return NULL if the entry is not found.

   Use _Py_hashtable_get_entry() to distinguish entry value equal to NULL
   and entry not found. */
PyAPI_FUNC(void*) _Py_hashtable_get(_Py_hashtable_t *ht, const void *key);


/* Remove a key and its associated value without calling key and value destroy
   functions.

   Return the removed value if the key was found.
   Return NULL if the key was not found. */
PyAPI_FUNC(void*) _Py_hashtable_steal(
    _Py_hashtable_t *ht,
    const void *key);


#ifdef __cplusplus
}
#endif
#endif   /* !Py_INTERNAL_HASHTABLE_H */
